"""Tests for behavior shared by every image."""

import json
import subprocess

import pytest

from conftest import _run_python, docker_run


# ----------------------------
#      Environment
# ----------------------------

# Invariants of every image; shell checks share one container per image via env_probe.
class TestEnvironment:
    def test_python_location(self, any_image, env_probe):
        assert env_probe(any_image)["which_python3"] == "/opt/conda/bin/python3"

    def test_python_version(self, any_image, env_probe):
        assert "3.12" in env_probe(any_image)["python_version"]

    def test_no_third_python(self, any_image, env_probe):
        """Only conda + Ubuntu python3; 0b374c0 dropped NVIDIA's third interpreter."""
        count = env_probe(any_image)["unique_python"]
        assert count == "2", f"expected conda + system python only, found {count}"

    def test_mamba(self, any_image, env_probe):
        assert env_probe(any_image)["which_mamba"] == "/opt/conda/bin/mamba"

    def test_conda(self, any_image, env_probe):
        assert env_probe(any_image)["conda_rc"] == "0", "conda not on PATH"

    def test_uv(self, any_image, env_probe):
        assert env_probe(any_image)["uv_rc"] == "0", "uv not on PATH"

    def test_r(self, any_image, env_probe):
        assert env_probe(any_image)["rscript_rc"] == "0", "Rscript not runnable"

    def test_tini(self, any_image, env_probe):
        assert env_probe(any_image)["which_tini"], "tini not on PATH"

    def test_git(self, any_image, env_probe):
        assert env_probe(any_image)["git_rc"] == "0", "git not runnable"

    def test_gh(self, any_image, env_probe):
        """GitHub CLI, installed via apt.txt so it reaches every image."""
        assert env_probe(any_image)["gh_rc"] == "0", "gh not runnable"

    def test_s5cmd(self, any_image, env_probe):
        assert env_probe(any_image)["s5cmd_rc"] == "0", "s5cmd not runnable"

    def test_home_dir_exists(self, any_image, env_probe):
        """HOME is created in the image; work/ and shared/ arrive as NFS mounts."""
        assert env_probe(any_image)["home_dir"] == "yes"

    def test_umask_is_000_in_interactive_shell(self, any_image, env_probe):
        """umask 000 from /etc/bash.bashrc -- interactive shells only, not kernels."""
        assert env_probe(any_image)["umask"] == "0000"

    def test_runs_as_root(self, any_image, env_probe):
        """85fbee4 removed jovyan; images run as root."""
        assert env_probe(any_image)["uid"] == "0"

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
#      AnnData / h5ad
# ----------------------------

class TestAnnData:
    """Regression guards for two anndata bugs an import test would miss."""

    def test_h5ad_write_with_string_obs_index(self, stack_image):
        """Save an AnnData with a string obs index (anndata 0.12.6 raised on this)."""
        script = """
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
rng = np.random.default_rng(0)
X = sp.csr_matrix(rng.random((10, 4), dtype=np.float32))
obs = pd.DataFrame({"cell": [f"c{i}" for i in range(10)]}).set_index("cell")
a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))
a.write_h5ad("/tmp/t.h5ad")
b = ad.read_h5ad("/tmp/t.h5ad")
assert b.shape == (10, 4), b.shape
assert list(b.obs_names[:2]) == ["c0", "c1"], list(b.obs_names[:2])
print("H5AD_OK")
"""
        result = _run_python(stack_image, script, timeout=180)
        assert "H5AD_OK" in result.stdout, f"h5ad round-trip failed:\n{result.stdout}"

    def test_layers_keys_are_strings(self, stack_image):
        """.layers must not yield a None key (anndata 0.13 exposes X as layers[None])."""
        script = """
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
rng = np.random.default_rng(0)
X = sp.csr_matrix(rng.random((10, 4), dtype=np.float32))
a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
               var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))
a.layers["counts"] = X.copy()
a.write_h5ad("/tmp/l.h5ad")
b = ad.read_h5ad("/tmp/l.h5ad")
keys = list(b.layers)
print("LAYER_KEYS", repr(keys))
print("LAYER_LEN", len(b.layers))
"""
        result = _run_python(stack_image, script, timeout=180)
        assert "LAYER_KEYS" in result.stdout, result.stdout
        keys_line = [l for l in result.stdout.splitlines() if l.startswith("LAYER_KEYS")][0]
        assert "None" not in keys_line, (
            f"anndata exposes a None layer key (X); iterating .layers will "
            f"silently touch X and sorted(keys()) raises TypeError -- {keys_line}"
        )
        len_line = [l for l in result.stdout.splitlines() if l.startswith("LAYER_LEN")][0]
        assert len_line.split()[1] == "1", f"expected exactly one layer -- {len_line}"


# ----------------------------
#      End-to-end workflows
# ----------------------------

# Exercise whole subsystems -- catches interop breakage imports don't.
class TestWorkflows:
    def test_single_cell_pipeline(self, stack_image):
        """scanpy pipeline must recover two planted clusters (anndata+sklearn+numba+leidenalg)."""
        script = """
import numpy as np, scipy.sparse as sp, anndata as ad, scanpy as sc
rng = np.random.default_rng(0)
counts = rng.poisson(0.3, (200, 300)).astype("float32")
counts[:100, :50] += rng.poisson(3.0, (100, 50)).astype("float32")
counts[100:, 50:100] += rng.poisson(3.0, (100, 50)).astype("float32")
a = ad.AnnData(sp.csr_matrix(counts))
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
sc.pp.pca(a, n_comps=20)
sc.pp.neighbors(a, n_neighbors=15)
sc.tl.leiden(a)
assert a.obsm["X_pca"].shape == (200, 20), a.obsm["X_pca"].shape
n = a.obs["leiden"].nunique()
assert n >= 2, f"planted 2 groups, leiden found {n}"
print("SCANPY_OK", n)
"""
        result = _run_python(stack_image, script, timeout=180)
        assert "SCANPY_OK" in result.stdout, f"scanpy pipeline failed:\n{result.stdout}"

    def test_parquet_cross_library_roundtrip(self, stack_image):
        """pandas writes a parquet; polars and duckdb must read it and agree."""
        script = """
import numpy as np, pandas as pd, polars as pl, duckdb
rng = np.random.default_rng(0)
df = pd.DataFrame({"i": range(500), "f": rng.random(500), "s": [f"x{i}" for i in range(500)]})
df.to_parquet("/tmp/t.parquet", index=False)
assert pl.read_parquet("/tmp/t.parquet").height == 500
n, mean = duckdb.sql("select count(*), avg(f) from '/tmp/t.parquet'").fetchone()
assert n == 500, n
assert abs(mean - df["f"].mean()) < 1e-9, (mean, df["f"].mean())
print("PARQUET_OK")
"""
        result = _run_python(stack_image, script, timeout=120)
        assert "PARQUET_OK" in result.stdout, f"parquet round-trip failed:\n{result.stdout}"


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
