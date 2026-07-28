# containers

Docker images maintained by the Briney Lab, published to Docker Hub under
[`brineylab`](https://hub.docker.com/u/brineylab). Images are built and pushed by
[`docker-publish.yml`](.github/workflows/docker-publish.yml) on each GitHub release,
tagged with both the release tag and `latest`.

## Images

| Image | Built on | Contents |
|---|---|---|
| `brineylab/base` | NVIDIA CUDA DL Base | OS tooling, the Python and R interpreters, conda/mamba, uv, s5cmd. No scientific, AI/ML, or R libraries. |
| `brineylab/datascience` | `ubuntu:24.04` | Scientific Python, R, and the NGS toolchain (CellRanger, dorado, IgDiscover, ...). |
| `brineylab/deeplearning` | NVIDIA CUDA DL Base | Everything in datascience minus NGS tools, plus PyTorch, JAX, HuggingFace, DeepSpeed, and flash-attn. |
| `brineylab/jupyterhub-base` | `brineylab/base` | Adds JupyterLab, Notebook, JupyterHub, and the R kernel. |
| `brineylab/jupyterhub-datascience` | `brineylab/datascience` | Adds JupyterLab, Notebook, and JupyterHub. |
| `brineylab/jupyterhub-deeplearning` | `brineylab/deeplearning` | Adds JupyterLab, Notebook, JupyterHub, and the GPU dashboard. |

The `base` images build on NVIDIA's CUDA image but contain no driver, so they run
on CPU-only hosts as well; the CUDA libraries are simply unused.

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
```

**To add a package, go to [`requirements/`](requirements).** It is always there,
for every image and every package manager:

| file | contents | used by |
|---|---|---|
| `apt.txt` | OS packages | base, datascience, deeplearning |
| `pip.txt` | scientific and biology Python packages | datascience, deeplearning |
| `ai-ml_pip.txt` | AI/ML packages | deeplearning |
| `r_conda.txt` / `r_cran.txt` | R packages | datascience, deeplearning |
| `jupyter_pip.txt` | Jupyter packages | all three `jupyterhub-*` images |

The `jupyterhub-*` images inherit everything above through their parent image, so
a package added to `apt.txt` reaches all six.

[`runtime/`](runtime) holds files copied into the images rather than installed:
`initial-condarc` (conda channels), the `start-notebook.py` and
`start-singleuser.py` entrypoints, `jupyter_server_config.py`,
`docker_healthcheck.py`, and `Rprofile.site`.

Which image is built from which parent is declared in each Dockerfile's `FROM`
line and in the workflow's `needs:`, not in this directory structure.

Builds run with the repository root as the Docker context, so `COPY` paths are
written relative to the root rather than to the Dockerfile.
