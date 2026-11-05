# containers

Docker images maintained by the Briney Lab, published to Docker Hub under
[`brineylab`](https://hub.docker.com/u/brineylab). Publishing a GitHub release builds the
tagged commit with [`docker-publish.yml`](.github/workflows/docker-publish.yml), tests every
image (deeplearning on a GPU too), and pushes them tagged with both the release tag and `latest`
once every test has passed. Builds run on the lab's self-hosted runners; [`AGENTS.md`](./AGENTS.md)
covers those, the hosted fallback, and when a release reuses an earlier build.

> Working in this repo, human or agent? See [`AGENTS.md`](./AGENTS.md) for who consumes these
> images, how builds are triggered, and the conventions.

## Images

| Image | Built on | Contents |
|---|---|---|
| `brineylab/base` | NVIDIA CUDA DL Base | OS tooling, the Python and R interpreters, conda/mamba, uv, s5cmd. No scientific, AI/ML, or R libraries. |
| `brineylab/datascience` | `ubuntu:24.04` | Scientific Python, R, Rust, mmseqs2, and the NGS toolchain (CellRanger, dorado, IgDiscover, ...). |
| `brineylab/deeplearning` | NVIDIA CUDA DL Base | Everything in datascience minus the NGS tools, plus PyTorch, JAX, HuggingFace, DeepSpeed, OpenMM, and flash-attn. |
| `brineylab/jupyterhub-base` | `brineylab/base` | Adds JupyterLab, Notebook, JupyterHub, and the R kernel. |
| `brineylab/jupyterhub-datascience` | `brineylab/datascience` | Adds JupyterLab, Notebook, and JupyterHub. |
| `brineylab/jupyterhub-deeplearning` | `brineylab/deeplearning` | Adds JupyterLab, Notebook, JupyterHub, and the GPU dashboard. |

The `base` images build on NVIDIA's CUDA image but contain no driver, so they run
on CPU-only hosts as well; the CUDA libraries are simply unused.

## arm64 images

`base`, `datascience` and `deeplearning` are also built for arm64, for the DGX Sparks (Slurm
partition `gb10`), as `brineylab/<image>:<tag>-arm64` and `:latest-arm64`. They come from the same
Dockerfiles, built and tested natively on niblet by
[`docker-publish-arm64.yml`](.github/workflows/docker-publish-arm64.yml), which a release starts
alongside the x86 build. An arm64 failure never holds up the x86 images. There are no arm64
`jupyterhub-*` images.

Slurm runs them as `.sqsh` files whose name carries `arm64`. Build those with
`slinky-config/tools/build-sqsh.sh` with `POD` set to a Spark worker pod, since `enroot import` takes
the architecture of the pod it runs in.

Four tools are **absent** on arm64; keep that work on x86 nodes:

| Tool | Why |
|---|---|
| `bcl2fastq2` | Illumina ships an amd64 `.deb` only, no source. |
| `bases2fastq` | Element's binary is x86-64 only. |
| `cellranger` | 10x Genomics ships x86-64 only. |
| IgDiscover | Its environment cannot be satisfied on aarch64. |

IgDiscover's case has scientific stakes. Two of its pins have no aarch64 build and exist to keep
its output correct: `igblast =1.19` (1.20+ adds AIRR columns and changes assignments) and
`muscle =3.8.1551` (MUSCLE 5 has an incompatible CLI). `flash` and `tinyalign` have no aarch64 conda
build either. Do not bump the pins to whatever aarch64 has; supporting IgDiscover on arm means
building those versions from source and validating against x86.

`dorado` on arm64 comes from ONT's CDN (the lab mirror holds only `linux-x64`), and `r-locfit`
comes from conda because its CRAN source build fails on aarch64, which would silently drop DESeq2.

## Rolling your own environment

`base` exists so you can install your own packages rather than inherit ours. It
ships the Python and R interpreters and their package managers, but no libraries:

```dockerfile
FROM brineylab/base

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache --system -r /tmp/requirements.txt

RUN Rscript -e "install.packages('data.table', repos = 'https://cran.rstudio.com')"
```

Or create a separate conda environment at runtime with `mamba create -n myenv ...`.

## Repository layout

Files are grouped by what they are, not by which image uses them. Each directory
under `images/` is named exactly as the image it publishes, which is also its job
name in the workflow.

```
requirements/   package lists — the thing you edit to add a package
runtime/        config and scripts baked into the images
images/<name>/  one Dockerfile per published image
tests/          pytest suite run against the built images
.github/        the build, release and constraints-bump workflows, and their actions and scripts
```

**To add a package, go to [`requirements/`](requirements).** It is always there,
for every image and every package manager:

| file | contents | used by |
|---|---|---|
| `apt.txt` | OS packages | base, datascience, deeplearning |
| `pip.txt` | scientific and biology Python packages | datascience, deeplearning |
| `ai-ml_pip.txt` | AI/ML packages | deeplearning |
| `datascience_pip.txt` | Python packages for datascience only | datascience |
| `r_conda.txt` / `r_cran.txt` | R packages | datascience, deeplearning |
| `conda.txt` | non-R conda packages (Rust, mmseqs2) | datascience, deeplearning |
| `jupyter_pip.txt` | Jupyter packages | all three `jupyterhub-*` images |
| `constraints.txt` | exact versions for every pip package above (generated) | every `uv pip install` |

The `jupyterhub-*` images inherit everything above through their parent image, so
a package added to `apt.txt` reaches all six.

**After editing a `*_pip.txt` file, run `requirements/compile-constraints.sh`** and commit
`constraints.txt` with it; a package missing from it installs unpinned. Every image installs
pip packages against that one file, so a shared package has the same version in every image, and
a cached layer stays valid exactly as long as its pins. `bump-pip-constraints.yml` reruns the
script with `--upgrade` every Monday and opens (or refreshes) a PR from `deps/pip-constraints`.

**After a package change, add a test that uses it**, not just imports it (`tests/`; see
`TestCompiledTools` in `test_base.py`). Run the suite against a local build with
`uvx --with pytest-xdist pytest tests/ -n 8 --dist loadscope --tag=<tag>`.

[`runtime/`](runtime) holds files copied into the images rather than installed:
`initial-condarc` (conda channels), the `start-notebook.py` and
`start-singleuser.py` entrypoints, `jupyter_server_config.py`,
`docker_healthcheck.py`, and `Rprofile.site`.

Which image is built from which parent is declared in each Dockerfile's `FROM`
line and in the workflow's `needs:`, not in this directory structure.

Builds run with the repository root as the Docker context, so `COPY` paths are
written relative to the root rather than to the Dockerfile.
