# containers

Docker images maintained by the Briney Lab, published to Docker Hub under
[`brineylab`](https://hub.docker.com/u/brineylab). Images are built and pushed by
[`docker-publish.yml`](.github/workflows/docker-publish.yml) on each GitHub release,
tagged with both the release tag and `latest`.

## Images

| Image | Built on | Contents |
|---|---|---|
| `brineylab/base` | NVIDIA CUDA DL Base | OS tooling, Python, conda/mamba, uv, s5cmd. No scientific, AI/ML, or R packages. |
| `brineylab/datascience` | `ubuntu:24.04` | Scientific Python, R, and the NGS toolchain (CellRanger, dorado, IgDiscover, ...). |
| `brineylab/deeplearning` | NVIDIA CUDA DL Base | Everything in datascience minus NGS tools, plus PyTorch, JAX, HuggingFace, DeepSpeed, and flash-attn. |
| `brineylab/jupyterhub-base` | `brineylab/base` | Adds JupyterLab, Notebook, and JupyterHub. |
| `brineylab/jupyterhub-datascience` | `brineylab/datascience` | Adds JupyterLab, Notebook, and JupyterHub. |
| `brineylab/jupyterhub-deeplearning` | `brineylab/deeplearning` | Adds JupyterLab, Notebook, JupyterHub, and the GPU dashboard. |

The `base` images build on NVIDIA's CUDA image but contain no driver, so they run
on CPU-only hosts as well; the CUDA libraries are simply unused. They provide a
Python kernel only, since R is not installed.

## Rolling your own environment

`base` exists so you can install your own packages rather than inherit ours:

```dockerfile
FROM brineylab/base

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache --system -r /tmp/requirements.txt
```

Or create a separate conda environment at runtime with `mamba create -n myenv ...`.

## Package lists

Packages live in [`requirements/`](requirements) rather than in the Dockerfiles:

- `apt.txt` — OS packages, used by every image
- `pip.txt` — scientific and biology Python packages (datascience, deeplearning)
- `ai-ml_pip.txt` — AI/ML packages (deeplearning only)
- `r_conda.txt` / `r_cran.txt` — R packages (datascience, deeplearning)
- `jupyter_pip.txt` — Jupyter packages (`jupyterhub-*` images only)
