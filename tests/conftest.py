"""Shared fixtures for container tests."""

import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--tag", default="latest", help="Docker image tag to test (default: latest)")


def docker_run(image: str, command: str, timeout: int = 60, gpus: bool = False) -> subprocess.CompletedProcess:
    """Run a command inside a Docker container and return the result.

    Note: stdout and stderr are merged so callers don't need to check both.
    """
    cmd = ["docker", "run", "--rm"]
    if gpus:
        cmd += ["--gpus", "all"]
    cmd += [image, "bash", "-c", command]
    return subprocess.run(cmd, capture_output=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)


@pytest.fixture(params=["brineylab/datascience", "brineylab/deeplearning"])
def base_image(request):
    """Parametrize tests across both base images."""
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"


@pytest.fixture(params=["brineylab/jupyterhub-datascience", "brineylab/jupyterhub-deeplearning"])
def jupyterhub_image(request):
    """Parametrize tests across both jupyterhub images."""
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"


@pytest.fixture(params=[
    "brineylab/datascience",
    "brineylab/deeplearning",
    "brineylab/jupyterhub-datascience",
    "brineylab/jupyterhub-deeplearning",
])
def any_image(request):
    """Parametrize tests across all images."""
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"
