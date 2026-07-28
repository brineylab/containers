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


MINIMAL_IMAGES = ["brineylab/base", "brineylab/jupyterhub-base"]
STACK_IMAGES = ["brineylab/datascience", "brineylab/deeplearning"]
STACK_JUPYTERHUB_IMAGES = [
    "brineylab/jupyterhub-datascience",
    "brineylab/jupyterhub-deeplearning",
]
JUPYTERHUB_IMAGES = ["brineylab/jupyterhub-base"] + STACK_JUPYTERHUB_IMAGES
ALL_IMAGES = MINIMAL_IMAGES + STACK_IMAGES + STACK_JUPYTERHUB_IMAGES


@pytest.fixture(params=STACK_IMAGES)
def stack_image(request):
    """Parametrize across the images carrying the shared scientific stack.

    Excludes the base images, which deliberately ship no domain packages.
    """
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"


@pytest.fixture(params=MINIMAL_IMAGES)
def minimal_image(request):
    """Parametrize across the images that ship interpreters but no libraries."""
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"


@pytest.fixture(params=JUPYTERHUB_IMAGES)
def jupyterhub_image(request):
    """Parametrize tests across all jupyterhub images."""
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"


@pytest.fixture(params=STACK_JUPYTERHUB_IMAGES)
def stack_jupyterhub_image(request):
    """Parametrize across jupyterhub images built on the scientific stack.

    Excludes jupyterhub-base, which inherits from base and therefore lacks
    anything provided by pip.txt.
    """
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"


@pytest.fixture(params=ALL_IMAGES)
def any_image(request):
    """Parametrize tests across every published image."""
    tag = request.config.getoption("--tag")
    return f"{request.param}:{tag}"
