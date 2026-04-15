"""Tests for both base images (datascience and deeplearning)."""

import pytest

from conftest import docker_run


# ----------------------------
#      Environment
# ----------------------------

class TestEnvironment:
    def test_python_location(self, base_image):
        result = docker_run(base_image, "which python3")
        assert result.returncode == 0
        assert "/opt/conda/bin/python3" in result.stdout

    def test_python_version(self, base_image):
        result = docker_run(base_image, "python3 --version")
        assert result.returncode == 0
        assert "3.12" in result.stdout

    def test_single_python(self, base_image):
        """Ensure only one Python is in PATH (conda Python)."""
        result = docker_run(base_image, "which -a python3 | head -1")
        assert result.returncode == 0
        assert "/opt/conda/bin/python3" in result.stdout

    def test_mamba(self, base_image):
        result = docker_run(base_image, "which mamba")
        assert result.returncode == 0
        assert "/opt/conda/bin/mamba" in result.stdout

    def test_conda(self, base_image):
        result = docker_run(base_image, "which conda")
        assert result.returncode == 0

    def test_uv(self, base_image):
        result = docker_run(base_image, "which uv")
        assert result.returncode == 0

    def test_r(self, base_image):
        result = docker_run(base_image, "Rscript --version")
        assert result.returncode == 0

    def test_tini(self, base_image):
        result = docker_run(base_image, "which tini")
        assert result.returncode == 0

    def test_git(self, base_image):
        result = docker_run(base_image, "git --version")
        assert result.returncode == 0

    def test_fix_permissions(self, base_image):
        result = docker_run(base_image, "test -x /usr/local/bin/fix-permissions")
        assert result.returncode == 0

    def test_conda_dir_exists(self, base_image):
        result = docker_run(base_image, "test -d /opt/conda")
        assert result.returncode == 0

    def test_work_dir_exists(self, base_image):
        result = docker_run(base_image, "test -d /home/jovyan/work")
        assert result.returncode == 0


# ----------------------------
#      Scientific stack
# ----------------------------

class TestScientificStack:
    @pytest.mark.parametrize("package", [
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scikit-image",
        "statsmodels",
        "bokeh",
        "dask",
        "numba",
        "h5py",
        "sympy",
        "sqlalchemy",
        "altair",
        "cython",
        "pyarrow",
        "tables",
        "openpyxl",
        "protobuf",
        "ipympl",
    ])
    def test_scientific_import(self, base_image, package):
        import_name = (
            package
            .replace("scikit-learn", "sklearn")
            .replace("scikit-image", "skimage")
            .replace("protobuf", "google.protobuf")
        )
        result = docker_run(base_image, f"python3 -c 'import {import_name}'")
        assert result.returncode == 0, f"Failed to import {package}: {result.stdout}"

    def test_numpy_version_gte_2(self, base_image):
        result = docker_run(base_image, "python3 -c 'import numpy; assert int(numpy.__version__.split(\".\")[0]) >= 2'")
        assert result.returncode == 0, f"Expected numpy >= 2: {result.stdout}"


# ----------------------------
#      Biology packages
# ----------------------------

class TestBiology:
    @pytest.mark.parametrize("package", [
        "scanpy",
        "scvelo",
        "bbknn",
        "leidenalg",
        "umap",
        "biopython",
        "parasail",
        "doubletdetection",
        "harmonypy",
        "scanorama",
        "scrublet",
        "logomaker",
        "dnachisel",
        "pyfamsa",
    ])
    def test_biology_import(self, base_image, package):
        import_name = package.replace("biopython", "Bio")
        result = docker_run(base_image, f"python3 -c 'import {import_name}'")
        assert result.returncode == 0, f"Failed to import {package}: {result.stdout}"

    def test_abutils(self, base_image):
        result = docker_run(base_image, "python3 -c 'import abutils; print(abutils.__version__)'")
        assert result.returncode == 0, f"Failed to import abutils: {result.stdout}"
        assert "0.6" in result.stdout, f"Expected abutils >= 0.6.0: {result.stdout}"

    def test_openmm(self, base_image):
        result = docker_run(base_image, "python3 -c 'import openmm; print(openmm.__version__)'")
        assert result.returncode == 0, f"Failed to import openmm: {result.stdout}"

    def test_pdbfixer(self, base_image):
        result = docker_run(base_image, "python3 -c 'import pdbfixer'")
        assert result.returncode == 0, f"Failed to import pdbfixer: {result.stdout}"

    def test_fastcluster_numpy2(self, base_image):
        """Verify fastcluster works with NumPy 2.x (was broken with pre-built wheels)."""
        result = docker_run(base_image, """python3 -c "
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

class TestUtilities:
    @pytest.mark.parametrize("package", [
        "duckdb",
        "polars",
        "paramiko",
        "pymongo",
        "pytest",
        "yaml",
        "tqdm",
        "humanize",
    ])
    def test_utility_import(self, base_image, package):
        result = docker_run(base_image, f"python3 -c 'import {package}'")
        assert result.returncode == 0, f"Failed to import {package}: {result.stdout}"


# ----------------------------
#      R packages
# ----------------------------

class TestR:
    def test_tidyverse(self, base_image):
        result = docker_run(base_image, 'Rscript -e "library(tidyverse); cat(\'OK\\n\')"')
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_biocmanager(self, base_image):
        result = docker_run(base_image, 'Rscript -e "library(BiocManager); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_irkernel(self, base_image):
        result = docker_run(base_image, 'Rscript -e "library(IRkernel); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_deseq2(self, base_image):
        result = docker_run(base_image, 'Rscript -e "library(DESeq2); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_biomart(self, base_image):
        result = docker_run(base_image, 'Rscript -e "library(biomaRt); cat(\'OK\\n\')"')
        assert result.returncode == 0

    def test_r_duckdb(self, base_image):
        result = docker_run(base_image, 'Rscript -e "library(duckdb); cat(\'OK\\n\')"', timeout=120)
        assert result.returncode == 0


# ----------------------------
#     Version consistency
# ----------------------------

class TestVersionConsistency:
    """Verify key packages have the same version across images."""

    @pytest.mark.parametrize("package", [
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
    ])
    def test_version_matches(self, request, package):
        tag = request.config.getoption("--tag")
        cmd = f"python3 -c 'import importlib.metadata; print(importlib.metadata.version(\"{package}\"))'"

        ds = docker_run(f"brineylab/datascience:{tag}", cmd)
        dl = docker_run(f"brineylab/deeplearning:{tag}", cmd)

        assert ds.returncode == 0, f"datascience failed: {ds.stdout}"
        assert dl.returncode == 0, f"deeplearning failed: {dl.stdout}"
        assert ds.stdout.strip() == dl.stdout.strip(), \
            f"{package}: datascience={ds.stdout.strip()} != deeplearning={dl.stdout.strip()}"
