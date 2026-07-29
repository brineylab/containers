"""Tests for behavior shared by every image."""

import json
import subprocess

import pytest

from conftest import docker_run


# ----------------------------
#      Environment
# ----------------------------

# These are invariants of every published image, including the minimal base
# images, so they run against any_image rather than just the stack images.
class TestEnvironment:
    def test_python_location(self, any_image):
        result = docker_run(any_image, "which python3")
        assert result.returncode == 0
        assert "/opt/conda/bin/python3" in result.stdout

    def test_python_version(self, any_image):
        result = docker_run(any_image, "python3 --version")
        assert result.returncode == 0
        assert "3.12" in result.stdout

    def test_no_third_python(self, any_image):
        """Exactly two interpreters: conda's and Ubuntu's system python3.

        0b374c0 switched deeplearning to CUDA DL Base to drop a *third* Python
        that NVIDIA's PyTorch image bundled. Ubuntu's /usr/bin/python3 is
        unavoidable and harmless as long as conda's wins on PATH, which
        test_python_location covers. Symlinks are resolved because /bin/python3
        and /usr/bin/python3 are the same file.
        """
        result = docker_run(
            any_image, "which -a python3 | xargs -n1 readlink -f | sort -u | wc -l"
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "2", \
            f"expected conda + system python only, found {result.stdout.strip()} interpreters"

    def test_mamba(self, any_image):
        result = docker_run(any_image, "which mamba")
        assert result.returncode == 0
        assert "/opt/conda/bin/mamba" in result.stdout

    def test_conda(self, any_image):
        result = docker_run(any_image, "which conda")
        assert result.returncode == 0

    def test_uv(self, any_image):
        result = docker_run(any_image, "which uv")
        assert result.returncode == 0

    def test_r(self, any_image):
        result = docker_run(any_image, "Rscript --version")
        assert result.returncode == 0

    def test_tini(self, any_image):
        result = docker_run(any_image, "which tini")
        assert result.returncode == 0

    def test_git(self, any_image):
        result = docker_run(any_image, "git --version")
        assert result.returncode == 0

    def test_gh(self, any_image):
        """GitHub CLI, installed via apt.txt so it reaches every image."""
        result = docker_run(any_image, "gh --version")
        assert result.returncode == 0

    def test_s5cmd(self, any_image):
        result = docker_run(any_image, "s5cmd version")
        assert result.returncode == 0

    def test_conda_dir_exists(self, any_image):
        result = docker_run(any_image, "test -d /opt/conda")
        assert result.returncode == 0

    def test_home_dir_exists(self, any_image):
        """HOME is created in the image; work/ and shared/ arrive as NFS mounts."""
        result = docker_run(any_image, "test -d /home/jovyan")
        assert result.returncode == 0

    def test_umask_is_000_in_interactive_shell(self, any_image):
        """umask 000 is set in /etc/bash.bashrc.

        Scope matters: bash sources that file only for interactive shells, so
        this covers terminal sessions but NOT files written by a notebook
        kernel, which inherits the server process umask instead. Group-writable
        notebook output on shared NFS depends on NB_UMASK in the singleuser pod
        spec, which no image test can verify.
        """
        result = docker_run(any_image, "bash -ic umask")
        assert result.returncode == 0
        assert "0000" in result.stdout

    def test_runs_as_root(self, any_image):
        """85fbee4 removed the jovyan user; these images always run as root."""
        result = docker_run(any_image, "id -u")
        assert result.returncode == 0
        assert result.stdout.strip() == "0"

    def test_entrypoint_is_tini(self, any_image):
        """tini reaps zombies; losing it from ENTRYPOINT would be silent."""
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{json .Config.Entrypoint}}", any_image],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert result.returncode == 0, result.stdout
        assert json.loads(result.stdout.strip()) == ["tini", "--"]


# ----------------------------
#      Scientific stack
# ----------------------------

# package name (the test id) -> module actually imported
SCIENTIFIC_PACKAGES = {
    "scipy": "scipy",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "statsmodels": "statsmodels",
    "bokeh": "bokeh",
    "dask": "dask",
    "numba": "numba",
    "h5py": "h5py",
    "sympy": "sympy",
    "sqlalchemy": "sqlalchemy",
    "altair": "altair",
    "cython": "cython",
    "pyarrow": "pyarrow",
    "tables": "tables",
    "openpyxl": "openpyxl",
    "protobuf": "google.protobuf",
    "ipympl": "ipympl",
}


class TestScientificStack:
    @pytest.mark.parametrize("package", list(SCIENTIFIC_PACKAGES))
    def test_scientific_import(self, stack_image, import_probe, package):
        results = import_probe(stack_image, SCIENTIFIC_PACKAGES.values())
        ok, detail = results.get(SCIENTIFIC_PACKAGES[package], (False, "not reported by probe"))
        assert ok, f"Failed to import {package}: {detail}"

    def test_numpy_version_gte_2(self, stack_image, version_probe):
        version = version_probe(stack_image, ["numpy"])["numpy"]
        assert int(version.split(".")[0]) >= 2, f"Expected numpy >= 2, got {version}"


# ----------------------------
#      Biology packages
# ----------------------------

BIOLOGY_PACKAGES = {
    "scanpy": "scanpy",
    "scvelo": "scvelo",
    "bbknn": "bbknn",
    "leidenalg": "leidenalg",
    "umap": "umap",
    "biopython": "Bio",
    "parasail": "parasail",
    "doubletdetection": "doubletdetection",
    "harmonypy": "harmonypy",
    "scanorama": "scanorama",
    "scrublet": "scrublet",
    "logomaker": "logomaker",
    "dnachisel": "dnachisel",
    "pyfamsa": "pyfamsa",
    # the lab's own ab[x] suite -- previously only abutils was covered
    "abstar": "abstar",
    "scab": "scab",
    # antibody numbering
    "anarci": "anarci",
    "abnumber": "abnumber",
}


class TestBiology:
    @pytest.mark.parametrize("package", list(BIOLOGY_PACKAGES))
    def test_biology_import(self, stack_image, import_probe, package):
        results = import_probe(stack_image, BIOLOGY_PACKAGES.values())
        ok, detail = results.get(BIOLOGY_PACKAGES[package], (False, "not reported by probe"))
        assert ok, f"Failed to import {package}: {detail}"

    def test_abutils_meets_pin(self, stack_image, version_probe):
        """pip.txt pins abutils>=0.6.0 (numpy 2 compatibility).

        Compares release tuples rather than substring-matching "0.6", which
        would reject a legitimate 1.x and accept an unrelated 10.6.
        """
        version = version_probe(stack_image, SHARED_PINNED_PACKAGES)["abutils"]
        assert version != "MISSING", "abutils not installed"
        parts = tuple(int(p) for p in version.split(".")[:3] if p.isdigit())
        assert parts >= (0, 6, 0), f"Expected abutils >= 0.6.0, got {version}"

    # Trastuzumab (Herceptin) heavy-chain variable domain -- real, published sequence.
    TRASTUZUMAB_VH = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
        "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    )

    def test_anarcii_numbers_real_antibody(self, stack_image):
        """Import alone proves nothing for a numbering tool -- actually number one."""
        script = f"""
from anarcii import Anarcii
m = Anarcii(seq_type='antibody', batch_size=1, cpu=True, ncpu=1, mode='accuracy', verbose=False)
res = list(m.number([('vh', '{self.TRASTUZUMAB_VH}')]).values())[0]
assert res['chain_type'] == 'H', res['chain_type']
print('POSITIONS', len(res['numbering']))
"""
        result = docker_run(stack_image, f"python3 -c \"{script}\"", timeout=180)
        assert result.returncode == 0, f"anarcii numbering failed: {result.stdout}"
        assert "POSITIONS 128" in result.stdout, result.stdout

    def test_abnumber_numbers_real_antibody(self, stack_image):
        """abnumber delegates to classic anarci, which shells out to hmmscan.

        Both packages import cleanly without HMMER present and only fail at
        runtime, so an import test is not enough here.
        """
        script = f"""
from abnumber import Chain
ch = Chain('{self.TRASTUZUMAB_VH}', scheme='imgt')
print('CDR3', ch.cdr3_seq)
"""
        result = docker_run(stack_image, f"python3 -c \"{script}\"", timeout=180)
        assert result.returncode == 0, f"abnumber numbering failed: {result.stdout}"

    def test_fastcluster_numpy2(self, stack_image):
        """Verify fastcluster works with NumPy 2.x (was broken with pre-built wheels)."""
        result = docker_run(stack_image, """python3 -c "
import numpy as np
import fastcluster
from scipy.spatial.distance import pdist
data = np.random.rand(50, 5)
Z = fastcluster.linkage(pdist(data), method='average')
assert Z.shape == (49, 4)
print('OK')
" """)
        assert result.returncode == 0, f"fastcluster test failed: {result.stdout}"
        assert "OK" in result.stdout


# ----------------------------
#      Utility packages
# ----------------------------

UTILITY_MODULES = [
    "duckdb",
    "polars",
    "paramiko",
    "pymongo",
    "pytest",
    "yaml",
    "tqdm",
    "humanize",
]


class TestUtilities:
    @pytest.mark.parametrize("module", UTILITY_MODULES)
    def test_utility_import(self, stack_image, import_probe, module):
        results = import_probe(stack_image, UTILITY_MODULES)
        ok, detail = results.get(module, (False, "not reported by probe"))
        assert ok, f"Failed to import {module}: {detail}"


# ----------------------------
#      R packages
# ----------------------------

class TestR:
    def test_tidyverse(self, stack_image):
        result = docker_run(stack_image, 'Rscript -e "library(tidyverse); cat(\'OK\\n\')"')
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_biocmanager(self, stack_image):
        result = docker_run(stack_image, 'Rscript -e "library(BiocManager); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_irkernel(self, stack_image):
        result = docker_run(stack_image, 'Rscript -e "library(IRkernel); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_deseq2(self, stack_image):
        result = docker_run(stack_image, 'Rscript -e "library(DESeq2); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_biomart(self, stack_image):
        result = docker_run(stack_image, 'Rscript -e "library(biomaRt); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_r_duckdb(self, stack_image):
        result = docker_run(stack_image, 'Rscript -e "library(duckdb); cat(\'OK\\n\')"', timeout=120)
        assert result.returncode == 0


# ----------------------------
#     Version consistency
# ----------------------------

SHARED_PINNED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "scanpy",
    "abutils",
    "fastcluster",
    "h5py",
    "seaborn",
    "dask",
    "numba",
    "openpyxl",
]


class TestVersionConsistency:
    """Verify key packages have the same version across images."""

    @pytest.mark.parametrize("package", SHARED_PINNED_PACKAGES)
    def test_version_matches(self, request, version_probe, package):
        tag = request.config.getoption("--tag")
        ds = version_probe(f"brineylab/datascience:{tag}", SHARED_PINNED_PACKAGES)
        dl = version_probe(f"brineylab/deeplearning:{tag}", SHARED_PINNED_PACKAGES)

        assert ds.get(package) != "MISSING", f"{package} not installed in datascience"
        assert dl.get(package) != "MISSING", f"{package} not installed in deeplearning"
        assert ds.get(package) == dl.get(package), \
            f"{package}: datascience={ds.get(package)} != deeplearning={dl.get(package)}"
