"""Tests specific to the datascience image."""

import pytest

from conftest import docker_run


@pytest.fixture
def image(request):
    tag = request.config.getoption("--tag")
    return f"brineylab/datascience:{tag}"


class TestNGSTools:
    def test_fastqc(self, image):
        result = docker_run(image, "fastqc --version")
        assert result.returncode == 0
        assert "FastQC" in result.stdout

    def test_cellranger(self, image):
        result = docker_run(image, "cellranger --version")
        assert result.returncode == 0

    def test_sickle(self, image):
        result = docker_run(image, "which sickle")
        assert result.returncode == 0

    def test_cutadapt(self, image):
        result = docker_run(image, "cutadapt --version")
        assert result.returncode == 0

    def test_bases2fastq(self, image):
        result = docker_run(image, "test -x /tools/bases2fastq")
        assert result.returncode == 0

    def test_dorado(self, image):
        result = docker_run(image, "which dorado")
        assert result.returncode == 0

    def test_pandaseq(self, image):
        result = docker_run(image, "which pandaseq")
        assert result.returncode == 0

    def test_bcl2fastq(self, image):
        result = docker_run(image, "which bcl2fastq")
        assert result.returncode == 0


class TestNGSPaths:
    def test_tools_dir(self, image):
        result = docker_run(image, "test -d /tools")
        assert result.returncode == 0

    def test_cellranger_in_path(self, image):
        result = docker_run(image, "echo $PATH | grep -q cellranger")
        assert result.returncode == 0

    def test_dorado_in_path(self, image):
        result = docker_run(image, "echo $PATH | grep -q dorado")
        assert result.returncode == 0


class TestIgDiscover:
    def test_igdiscover_env_exists(self, image):
        result = docker_run(image, "mamba env list | grep igdiscover")
        assert result.returncode == 0

    def test_igdiscover_runs(self, image):
        result = docker_run(image, "mamba run -n igdiscover igdiscover --help")
        assert result.returncode == 0


# ----------------------------
#      Excluded packages
# ----------------------------

class TestExcludedPackages:
    """openmm moved to deeplearning only in 30795ba.

    These guard against it drifting back into the shared pip.txt.
    """

    @pytest.mark.parametrize("module", ["openmm", "pdbfixer"])
    def test_module_absent(self, image, module):
        result = docker_run(image, f"python3 -c 'import {module}'")
        assert result.returncode != 0, \
            f"{module} should not be in the datascience image, but imported successfully"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known issue: pip.txt -> abnumber -> anarcii pulls torch, and pip "
            "resolves the default CUDA wheel. That puts ~1.1 GB of torch plus "
            "~2.7 GB of nvidia/* CUDA libraries into the CPU-only image. Fix is "
            "to install torch from the PyTorch CPU index in this image. Remove "
            "this marker once that lands."
        ),
    )
    def test_torch_is_cpu_build_if_present(self, image):
        """datascience runs on CPU nodes, so any torch here should be the CPU wheel."""
        result = docker_run(image, "python3 -c 'import torch; print(torch.__version__)'")
        if result.returncode != 0:
            pytest.skip("torch not installed, nothing to check")
        assert "+cu" not in result.stdout, \
            f"CUDA torch build in the CPU image: {result.stdout.strip()}"
