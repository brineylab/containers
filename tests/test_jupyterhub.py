"""Tests for JupyterHub images."""

import pytest

from conftest import docker_run


# ----------------------------
#      Jupyter core
# ----------------------------

class TestJupyter:
    def test_jupyterlab(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "jupyter lab --version")
        assert result.returncode == 0

    def test_jupyterhub(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "jupyterhub --version")
        assert result.returncode == 0

    def test_notebook(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "jupyter notebook --version")
        assert result.returncode == 0

    def test_singleuser(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "which jupyterhub-singleuser")
        assert result.returncode == 0
        assert "/opt/conda/bin/jupyterhub-singleuser" in result.stdout

    def test_server_config(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "test -f /etc/jupyter/jupyter_server_config.py")
        assert result.returncode == 0

    def test_healthcheck(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "test -f /etc/jupyter/docker_healthcheck.py")
        assert result.returncode == 0

    def test_start_notebook(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "test -f /usr/local/bin/start-notebook.py")
        assert result.returncode == 0

    def test_start_singleuser(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "test -f /usr/local/bin/start-singleuser.py")
        assert result.returncode == 0

    def test_rprofile(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "test -f /opt/conda/lib/R/etc/Rprofile.site")
        assert result.returncode == 0


# ----------------------------
#      Kernels
# ----------------------------

class TestKernels:
    def test_python_kernel(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "jupyter kernelspec list")
        assert result.returncode == 0
        assert "python3" in result.stdout

    def test_r_kernel(self, jupyterhub_image):
        result = docker_run(jupyterhub_image, "jupyter kernelspec list")
        assert result.returncode == 0
        assert "ir" in result.stdout


# ----------------------------
#      Lab extensions
# ----------------------------

class TestLabExtensions:
    @pytest.mark.parametrize("extension", [
        "jupyterlab_code_formatter",
        "@jupyterlab/git",
        "@jupyterlab/github",
        "@marimo-team/jupyter-extension",
        "jupyter-matplotlib",
        "@jupyter-widgets/jupyterlab-manager",
        "@jupyterhub/jupyter-server-proxy",
    ])
    def test_lab_extension(self, jupyterhub_image, extension):
        result = docker_run(jupyterhub_image, "jupyter labextension list")
        assert extension in result.stdout, \
            f"{extension} not found in labextension list"

    def test_nvdashboard(self, request):
        tag = request.config.getoption("--tag")
        result = docker_run(f"brineylab/jupyterhub-deeplearning:{tag}", "jupyter labextension list")
        assert "nvdashboard" in result.stdout


# ----------------------------
#      Server extensions
# ----------------------------

class TestServerExtensions:
    @pytest.mark.parametrize("extension", [
        "jupyterlab",
        "jupyterlab_code_formatter",
        "jupyterlab_git",
        "nbgitpuller",
        "notebook",
        "jupyter_server_proxy",
    ])
    def test_server_extension(self, jupyterhub_image, extension):
        result = docker_run(jupyterhub_image, "jupyter server extension list")
        assert extension in result.stdout, \
            f"{extension} not found in server extension list"

    def test_nvdashboard_server(self, request):
        tag = request.config.getoption("--tag")
        result = docker_run(f"brineylab/jupyterhub-deeplearning:{tag}", "jupyter server extension list")
        assert "jupyterlab_nvdashboard" in result.stdout
