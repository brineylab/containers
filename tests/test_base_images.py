"""Tests for the minimal base images.

base and jupyterhub-base ship the Python and R interpreters plus package
managers, but deliberately none of the scientific, AI/ML, or R libraries the
datascience and deeplearning images carry. These tests assert that emptiness --
it is the whole point of the images, and it is what silently regresses if a
package leaks into a shared requirements file.
"""

import pytest

from conftest import docker_run


class TestInterpretersPresent:
    def test_r_interpreter(self, minimal_image):
        result = docker_run(minimal_image, "Rscript -e 'cat(R.version.string)'")
        assert result.returncode == 0
        assert "R version" in result.stdout

    def test_r_install_packages_available(self, minimal_image):
        """Users are expected to install their own R packages."""
        result = docker_run(minimal_image, "Rscript -e 'cat(is.function(install.packages))'")
        assert result.returncode == 0
        assert "TRUE" in result.stdout

    def test_nvcc(self, minimal_image):
        """The CUDA toolkit is present even though no driver is baked in."""
        result = docker_run(minimal_image, "nvcc --version")
        assert result.returncode == 0


class TestNoDomainPackages:
    @pytest.mark.parametrize("module", [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scanpy",
        "abutils",
        "torch",
        "transformers",
    ])
    def test_python_module_absent(self, minimal_image, module):
        result = docker_run(minimal_image, f"python3 -c 'import {module}'")
        assert result.returncode != 0, \
            f"{module} leaked into a base image, which should ship no libraries"

    @pytest.mark.parametrize("package", ["tidyverse", "ggplot2", "BiocManager"])
    def test_r_package_absent(self, minimal_image, package):
        result = docker_run(
            minimal_image,
            f"Rscript -e 'quit(status = as.integer(requireNamespace(\"{package}\", quietly = TRUE)))'",
        )
        assert result.returncode == 0, \
            f"R package {package} leaked into a base image"

    def test_no_ngs_tools(self, minimal_image):
        result = docker_run(minimal_image, "test -d /tools")
        assert result.returncode != 0, "NGS tools directory should not exist in a base image"


class TestBaseHasNoJupyter:
    """Only the jupyterhub-* variant should carry Jupyter."""

    def test_jupyter_absent(self, request):
        tag = request.config.getoption("--tag")
        result = docker_run(f"brineylab/base:{tag}", "which jupyter")
        assert result.returncode != 0, "jupyter should only be in the jupyterhub images"
