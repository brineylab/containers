"""Tests specific to the deeplearning image."""

import subprocess

import pytest

from conftest import docker_run


@pytest.fixture
def image(request):
    tag = request.config.getoption("--tag")
    return f"brineylab/deeplearning:{tag}"


def _gpu_available() -> bool:
    """Check if GPU is available on the host."""
    result = subprocess.run(["nvidia-smi"], capture_output=True)
    return result.returncode == 0


# ----------------------------
#      AI/ML packages
# ----------------------------

class TestAIML:
    @pytest.mark.parametrize("package", [
        "torch",
        "lightning",
        "deepspeed",
        "wandb",
        "jax",
        "flax",
        "optax",
        "equinox",
        "chex",
        "keras",
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
    ])
    def test_aiml_import(self, image, package):
        env = "KERAS_BACKEND=torch " if package == "keras" else ""
        result = docker_run(image, f"{env}python3 -c 'import {package}'")
        assert result.returncode == 0, f"Failed to import {package}: {result.stdout}"

    def test_torch_cuda_build(self, image):
        result = docker_run(image, "python3 -c 'import torch; print(torch.__version__)'")
        assert result.returncode == 0
        assert "cu130" in result.stdout, f"Expected CUDA 13 build, got: {result.stdout.strip()}"

    def test_deepspeed_ops(self, image):
        """Verify libaio is available for DeepSpeed async I/O."""
        result = docker_run(image, "test -f /usr/lib/x86_64-linux-gnu/libaio.so || test -f /usr/lib/x86_64-linux-gnu/libaio.so.1")
        assert result.returncode == 0, "libaio not found — DeepSpeed async I/O will fail"


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
    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_torch_cuda(self, image):
        result = docker_run(image, "python3 -c 'import torch; assert torch.cuda.is_available()'", gpus=True)
        assert result.returncode == 0, f"torch CUDA not available: {result.stdout}"

    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_torch_gpu_name(self, image):
        result = docker_run(image, "python3 -c 'import torch; print(torch.cuda.get_device_name(0))'", gpus=True)
        assert result.returncode == 0

    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_jax_gpu(self, image):
        result = docker_run(
            image,
            'python3 -c "import jax; devs = jax.devices(); assert any(d.platform == \'gpu\' for d in devs)"',
            gpus=True,
        )
        assert result.returncode == 0, f"JAX GPU not available: {result.stdout}"

    @pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
    def test_torch_tensor_on_gpu(self, image):
        result = docker_run(
            image,
            "python3 -c 'import torch; x = torch.randn(10).cuda(); print(x.device)'",
            gpus=True,
        )
        assert result.returncode == 0
        assert "cuda" in result.stdout
