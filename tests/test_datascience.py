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
        result = docker_run(image, "test -f /tools/bases2fastq")
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
