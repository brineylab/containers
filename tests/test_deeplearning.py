"""Tests specific to the deeplearning image."""

import subprocess

import pytest

from conftest import _run_python, docker_run


@pytest.fixture
def image(request):
    tag = request.config.getoption("--tag")
    return f"brineylab/deeplearning:{tag}"


def _gpu_available() -> bool:
    """Whether the host has a usable GPU.

    subprocess.run raises OSError (FileNotFoundError) when nvidia-smi is absent
    rather than returning non-zero. This runs at collection time inside a
    skipif decorator, so an unhandled raise aborts collection of the whole file
    and the suite reports an error without running a single test. CI runners
    have no nvidia-smi at all, which is exactly that case.
    """
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
    except OSError:
        return False
    return result.returncode == 0


# ----------------------------
#      AI/ML packages
# ----------------------------

AIML_MODULES = [
    "torch",
    "lightning",
    "deepspeed",
    "wandb",
    "jax",
    "flax",
    "optax",
    "equinox",
    "chex",
    "accelerate",
    "transformers",
    "datasets",
    "diffusers",
    "peft",
    "evaluate",
    "optimum",
    "gradio",
    "tree",
    "ml_collections",
    "treescope",
]


class TestAIML:
    @pytest.mark.parametrize("module", AIML_MODULES)
    def test_aiml_import(self, image, import_probe, module):
        results = import_probe(image, AIML_MODULES)
        ok, detail = results.get(module, (False, "not reported by probe"))
        assert ok, f"Failed to import {module}: {detail}"

    def test_keras_import(self, image):
        """Separate from the batch: keras needs a backend selected up front."""
        result = docker_run(image, "KERAS_BACKEND=torch python3 -c 'import keras'", timeout=120)
        assert result.returncode == 0, f"Failed to import keras: {result.stdout}"

    def test_torch_cuda_build(self, image):
        result = docker_run(image, "python3 -c 'import torch; print(torch.__version__)'")
        assert result.returncode == 0
        assert "cu130" in result.stdout, f"Expected CUDA 13 build, got: {result.stdout.strip()}"

    def test_libaio_present(self, image):
        """libaio must be present for DeepSpeed async I/O.

        Renamed from test_deepspeed_ops: this only checks the shared library
        exists, it does not build or load any DeepSpeed op.
        """
        result = docker_run(image, "test -f /usr/lib/x86_64-linux-gnu/libaio.so || test -f /usr/lib/x86_64-linux-gnu/libaio.so.1")
        assert result.returncode == 0, "libaio not found — DeepSpeed async I/O will fail"

    def test_flash_attn(self, image):
        """Added in #55, built from source against the installed torch.

        A source build against the wrong torch is the most fragile thing in
        this image, and it was previously untested.
        """
        result = docker_run(
            image,
            "python3 -c 'import flash_attn; print(flash_attn.__version__)'",
            timeout=120,
        )
        assert result.returncode == 0, f"Failed to import flash_attn: {result.stdout}"

    def test_compiler_compat_ld_moved(self, image):
        """40e3b07 moves conda's ld aside so DeepSpeed JIT uses the system linker."""
        result = docker_run(image, "test -f /opt/conda/compiler_compat/ld.bak && ! test -f /opt/conda/compiler_compat/ld")
        assert result.returncode == 0, "conda compiler_compat/ld was not moved aside"


# ----------------------------
#      NVIDIA environment
# ----------------------------

class TestNVIDIA:
    def test_nvidia_visible_devices(self, image):
        result = docker_run(image, "echo $NVIDIA_VISIBLE_DEVICES")
        assert "all" in result.stdout

    def test_nvidia_driver_capabilities(self, image):
        result = docker_run(image, "echo $NVIDIA_DRIVER_CAPABILITIES")
        assert "compute" in result.stdout

    def test_cuda_available(self, image):
        result = docker_run(image, "nvcc --version")
        assert result.returncode == 0
        assert "13." in result.stdout, f"Expected CUDA 13.x: {result.stdout}"


# ----------------------------
#      GPU tests
# ----------------------------

class TestGPU:
    @pytest.mark.gpu
    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_torch_cuda(self, image):
        result = docker_run(image, "python3 -c 'import torch; assert torch.cuda.is_available()'", gpus=True)
        assert result.returncode == 0, f"torch CUDA not available: {result.stdout}"

    @pytest.mark.gpu
    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_torch_gpu_name(self, image):
        result = docker_run(image, "python3 -c 'import torch; print(torch.cuda.get_device_name(0))'", gpus=True)
        assert result.returncode == 0

    @pytest.mark.gpu
    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_jax_gpu(self, image):
        result = docker_run(
            image,
            'python3 -c "import jax; devs = jax.devices(); assert any(d.platform == \'gpu\' for d in devs)"',
            gpus=True,
        )
        assert result.returncode == 0, f"JAX GPU not available: {result.stdout}"

    @pytest.mark.gpu
    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_torch_tensor_on_gpu(self, image):
        result = docker_run(
            image,
            "python3 -c 'import torch; x = torch.randn(10).cuda(); print(x.device)'",
            gpus=True,
        )
        assert result.returncode == 0
        assert "cuda" in result.stdout


# ----------------------------
#      Structural biology
# ----------------------------

# openmm and pdbfixer moved to the deeplearning image only in 30795ba.
# test_datascience.py asserts the matching absence.
class TestStructuralBiology:
    def test_openmm(self, image):
        result = docker_run(image, "python3 -c 'import openmm; print(openmm.__version__)'")
        assert result.returncode == 0, f"Failed to import openmm: {result.stdout}"

    def test_pdbfixer(self, image):
        result = docker_run(image, "python3 -c 'import pdbfixer'")
        assert result.returncode == 0, f"Failed to import pdbfixer: {result.stdout}"


# ----------------------------
#      Training workflows
# ----------------------------

# A few real CPU steps -- catches broken torch/transformers pairings imports don't.
# All tiny and network-free.
class TestTraining:
    def test_torch_trains_a_few_steps(self, image):
        """A small MLP must drive its loss down over 50 CPU steps."""
        script = """
import math, torch, torch.nn as nn
torch.manual_seed(0)
X = torch.randn(256, 20); w = torch.randn(20, 1)
y = X @ w + 0.1 * torch.randn(256, 1)
model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for _ in range(50):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(X), y)
    loss.backward(); opt.step()
    losses.append(loss.item())
assert all(math.isfinite(x) for x in losses), "loss went non-finite"
assert losses[-1] < losses[0] * 0.5, f"did not learn: {losses[0]:.3f} -> {losses[-1]:.3f}"
print("TORCH_OK")
"""
        result = _run_python(image, script, timeout=180)
        assert "TORCH_OK" in result.stdout, f"torch training failed:\n{result.stdout}"

    def test_transformers_masked_lm_step(self, image):
        """Tiny masked-LM from config does a training step -- the AbLM path in miniature."""
        script = """
import torch
from transformers import BertConfig, BertForMaskedLM
cfg = BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                 num_attention_heads=2, intermediate_size=64, max_position_embeddings=32)
model = BertForMaskedLM(cfg)
ids = torch.randint(0, 64, (2, 16))
out = model(input_ids=ids, labels=ids.clone())
out.loss.backward()
grad = model.bert.embeddings.word_embeddings.weight.grad
assert torch.isfinite(out.loss) and out.loss.item() > 0, out.loss
assert grad is not None and torch.isfinite(grad).all(), "no/invalid embedding grad"
print("TRANSFORMERS_OK")
"""
        result = _run_python(image, script, timeout=180)
        assert "TRANSFORMERS_OK" in result.stdout, f"transformers step failed:\n{result.stdout}"

    def test_jax_value_and_grad(self, image):
        """JAX autodiff on CPU must match the analytic gradient of sin(x)^2."""
        script = """
import math, jax, jax.numpy as jnp
val, grad = jax.value_and_grad(lambda x: jnp.sum(jnp.sin(x) ** 2))(jnp.arange(5.0))
assert math.isfinite(float(val))
assert all(abs(float(grad[i]) - math.sin(2 * i)) < 1e-4 for i in range(5)), grad
print("JAX_OK")
"""
        result = _run_python(image, script, timeout=180)
        assert "JAX_OK" in result.stdout, f"jax grad failed:\n{result.stdout}"
