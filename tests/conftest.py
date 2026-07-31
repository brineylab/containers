"""Shared fixtures for container tests.

Container startup dominates runtime, so repeated commands go through `cached_run`
and grouped imports through `import_probe` -- one container, one test per package.
"""

import base64
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


def _run_python(image: str, script: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a multi-line Python script in a container.

    Base64 so the script survives the bash -c layer regardless of quoting.
    """
    encoded = base64.b64encode(script.encode()).decode()
    return docker_run(image, f"echo {encoded} | base64 -d | python3", timeout=timeout)


@pytest.fixture(scope="session")
def cached_run():
    """docker_run memoized on (image, command) for the session."""
    cache: dict = {}

    def run(image: str, command: str, **kwargs) -> subprocess.CompletedProcess:
        key = (image, command)
        if key not in cache:
            cache[key] = docker_run(image, command, **kwargs)
        return cache[key]

    return run


@pytest.fixture(scope="session")
def import_probe():
    """Import a group of modules in ONE container. Returns {module: (ok, detail)}."""
    cache: dict = {}

    def probe(image: str, modules) -> dict:
        modules = list(modules)
        key = (image, tuple(modules))
        if key not in cache:
            script = (
                "import importlib\n"
                f"for m in {modules!r}:\n"
                "    try:\n"
                "        importlib.import_module(m)\n"
                "        print('OK', m)\n"
                "    except BaseException as e:\n"
                "        print('FAIL', m, type(e).__name__, str(e).replace(chr(10), ' '))\n"
            )
            result = _run_python(image, script)
            parsed = {}
            for line in result.stdout.splitlines():
                parts = line.split(None, 2)
                if len(parts) >= 2 and parts[0] in ("OK", "FAIL"):
                    parsed[parts[1]] = (parts[0] == "OK", parts[2] if len(parts) > 2 else "")
            if not parsed:  # probe never ran; don't report every module absent
                pytest.fail(
                    f"import probe produced no parsable output for {image}\n{result.stdout}"
                )
            cache[key] = parsed
        return cache[key]

    return probe


@pytest.fixture(scope="session")
def version_probe():
    """Read installed versions for a group of packages in ONE container."""
    cache: dict = {}

    def probe(image: str, packages) -> dict:
        packages = list(packages)
        key = (image, tuple(packages))
        if key not in cache:
            script = (
                "from importlib.metadata import version, PackageNotFoundError\n"
                f"for p in {packages!r}:\n"
                "    try:\n"
                "        print('VER', p, version(p))\n"
                "    except PackageNotFoundError:\n"
                "        print('VER', p, 'MISSING')\n"
            )
            result = _run_python(image, script)
            # 'VER' prefix so a stray two-word warning can't parse in
            parsed = {}
            for line in result.stdout.splitlines():
                parts = line.split(None, 2)
                if len(parts) == 3 and parts[0] == "VER":
                    parsed[parts[1]] = parts[2]
            if not parsed:
                pytest.fail(
                    f"version probe produced no parsable output for {image}\n{result.stdout}"
                )
            cache[key] = parsed
        return cache[key]

    return probe


# Everything TestEnvironment checks, emitted as key<TAB>value, one container per image.
_ENV_PROBE_SCRIPT = r"""
emit() { printf '%s\t%s\n' "$1" "$2"; }
emit which_python3  "$(which python3)"
emit python_version "$(python3 --version 2>&1)"
emit unique_python  "$(which -a python3 | xargs -n1 readlink -f | sort -u | wc -l | tr -d '[:space:]')"
emit which_mamba    "$(which mamba)"
emit conda_rc       "$(which conda >/dev/null 2>&1; echo $?)"
emit uv_rc          "$(which uv >/dev/null 2>&1; echo $?)"
emit rscript_rc     "$(Rscript --version >/dev/null 2>&1; echo $?)"
emit which_tini     "$(which tini)"
emit git_rc         "$(git --version >/dev/null 2>&1; echo $?)"
emit gh_rc          "$(gh --version >/dev/null 2>&1; echo $?)"
emit s5cmd_rc       "$(s5cmd version >/dev/null 2>&1; echo $?)"
emit home_dir       "$(test -d /home/jovyan && echo yes || echo no)"
emit umask          "$(bash -ic umask 2>/dev/null | tr -d '[:space:]')"
emit uid            "$(id -u)"
"""


@pytest.fixture(scope="session")
def env_probe():
    """Run every environment check in ONE container per image. Returns {key: value}."""
    cache: dict = {}

    def probe(image: str) -> dict:
        if image not in cache:
            result = docker_run(image, _ENV_PROBE_SCRIPT, timeout=120)
            env = {}
            for line in result.stdout.splitlines():
                if "\t" in line:
                    k, v = line.split("\t", 1)
                    env[k] = v
            if "uid" not in env:  # last line emitted; absent means script died early
                pytest.fail(f"env probe incomplete for {image}\n{result.stdout}")
            cache[image] = env
        return cache[image]

    return probe


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
