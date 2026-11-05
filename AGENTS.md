# AGENTS.md

Entry point for AI agents (Claude Code, Codex, Cursor, Aider, etc.) working in this repo. Read
this first, then [`README.md`](./README.md), which is accurate and explains the layout properly.

## What this repo is

The six Docker images the lab publishes to Docker Hub under
[`brineylab`](https://hub.docker.com/u/brineylab). They are what lab members actually compute in:
the `jupyterhub-*` images back the notebook servers on all three JupyterHub clusters, and
`datascience` / `deeplearning` are converted to SquashFS and run as Slurm jobs on the Slinky
clusters (see `brineylab/slinky-config`, `tools/build-sqsh.sh`).

An image change therefore reaches people two ways, on different schedules — a hub picks it up when
its pod is recreated, a Slurm job when someone rebuilds the `.sqsh`. There is no single "deploy"
that updates everyone.

## Where to look for what

| If you need... | Read |
|---|---|
| The image list, layout, and how to roll your own | [`README.md`](./README.md) |
| **To add a package** | [`requirements/`](./requirements/) — always, for every image and package manager |
| What gets baked in but not installed | [`runtime/`](./runtime/) |
| Which image builds from which parent | each `images/<name>/Dockerfile` `ARG BASE_IMG`, and the workflow's `needs:` |
| Build, test and publish | `.github/workflows/docker-publish.yml`, with `.github/actions/` (build, tests) |
| How a release starts | `.github/workflows/release.yml`, `.github/scripts/release-reuse-check.sh` |
| pip versions | `requirements/constraints.txt`, from `requirements/compile-constraints.sh` |
| The machines the jobs run on | `brineylab/ansible`, `roles/actions_runner` (legolas, flint) |

## How builds happen

**Images publish on a GitHub release**, not on merge — tagged with both the release tag and
`latest`. A merged PR changes nothing that anyone is running until a release is cut. Nothing builds
on push or on a PR. A manual run is a `workflow_dispatch` of `docker-publish.yml` with `tag`,
`use_cache`, `push` and `hosted` inputs; on any branch but `main` it gets no runner unless
`hosted=true`.

**Builds run on the lab's own runners.** The six builds and the CPU test jobs run on legolas's
`containers` pool (4 runners, one 160 GiB memory cap between them). `gpu-test` runs on flint's
`gpu` pool (an RTX 5070 Ti) and gets deeplearning's image through legolas's CI registry.
flash-attn compiles with `MAX_JOBS=8` there and 2 on hosted runners. GitHub never moves a job to
another kind of runner: if legolas or flint is down, jobs queue for up to 24 h. The way round is a
dispatch with `hosted=true`, which builds on GitHub's xlarge runners, skips `gpu-test`, and pushes
without the test gate below.

**A release builds the tagged commit, which must be on `main`.** `release.yml` dispatches
`docker-publish.yml` on `main` (the only ref the runner groups admit) with the tag; every job checks
the tag out. The build is uncached, unless `main` already built and tested that exact commit
uncached on legolas within 7 days, in which case those layers are reused and the release publishes
the tested images (`.github/scripts/release-reuse-check.sh`; the run's summary says which). Runs
queue rather than cancel each other.

GitHub runs the `release.yml` that is in the tagged commit, not `main`'s, so a change to it only
applies to releases of later commits. To publish without a second full build: dispatch
`docker-publish.yml` on `main` with `use_cache=false` and `push=false`, wait for it to pass, then
release that same commit within 7 days. If a release still starts the wrong build, cancel it (on
legolas only `publish` pushes) and dispatch `docker-publish.yml` on `main` with the tag yourself.

**On legolas nothing reaches Docker Hub until every test passed.** Each image family is tested
once its jupyterhub image is built (`test-base`, `test-datascience`, `test-deeplearning`),
`test-cross` compares shared package versions, `gpu-test` runs on flint, and `publish` pushes only
after all of them. `pytest --images a,b` runs just the tests that use those images. With
`hosted=true` the builds push as they finish, as before, since hosted jobs share no image store.

**arm64 is a separate workflow.** `docker-publish-arm64.yml` builds `base`, `datascience` and
`deeplearning` natively on niblet (a DGX Spark, also slinky-prod's `gb10` node; runner group
`arm64`, one runner capped at 48 GiB, alongside Slurm jobs), runs the same tests including the GPU
ones on the GB10, and pushes `<tag>-arm64` and `latest-arm64` only if they all passed. It has its
own concurrency group and no x86 job waits on it, so a failed or queued arm build never holds up an
x86 release; `release.yml` dispatches both. Tests for tools absent on arm carry `x86_only`. There is
no hosted fallback for arm.

The build graph is expressed in the workflow's `needs:`, so `base` → `jupyterhub-base`,
`datascience` → `jupyterhub-datascience`, `deeplearning` → `jupyterhub-deeplearning`. Builds run
with the **repository root** as the Docker context, so `COPY` paths are written relative to the
root, not to the Dockerfile.

## Tests

`pytest` against built images in `tests/`: `test_base.py` covers what every stack image shares,
`test_base_images.py` the two minimal images, then `test_datascience.py`, `test_deeplearning.py`
and `test_jupyterhub.py`. Each test is a `docker run`, so they parallelize:
`uvx --with pytest-xdist pytest tests/ -n 8 --dist loadscope --tag=<tag>`. `--images a,b` keeps only
the tests that use those images; a test that names an image directly, rather than through an
image fixture, needs `@pytest.mark.images` so `--images` can place it.

The `gpu` marker means "requires a GPU on the host": skipped without one, or failed with
`--require-gpu`, which `gpu-test` sets. A CPU-only run has not exercised CUDA. `gpu-test` covers
flint's sm120 GPU and 580 driver only, not the sm80/90/100 kernels the other lab GPUs use.

## Conventions

**Add packages in `requirements/`, not in a Dockerfile.** The split is by package manager and
audience (`apt.txt`, `pip.txt`, `datascience_pip.txt`, `ai-ml_pip.txt`, `r_conda.txt`, `conda.txt`,
`r_cran.txt`, `jupyter_pip.txt`), and the `jupyterhub-*` images inherit through their parents —
so a line added to `apt.txt` reaches all six images. That inheritance is the reason to resist
adding a package "just for one image".

**pip versions come from `requirements/constraints.txt`.** Regenerate it with
`requirements/compile-constraints.sh` after editing any `*_pip.txt`; never hand-edit it. The weekly
`bump-pip-constraints.yml` PR (branch `deps/pip-constraints`) is untested until someone dispatches
`docker-publish.yml` on it with `push=false`, and that branch must be in the runner groups' allowed
workflows first (org settings, runner groups `containers` and `gpu`).

**Pin what you can reason about.** These images are the reproducibility floor for lab analyses;
an unpinned dependency changes what a rebuilt image does without any commit here.

**Commit messages use Conventional Commits** — `type(scope): subject`, types `feat`/`fix`/`docs`/
`refactor`/`test`/`chore`, scope usually the image or `requirements`. Work on a branch and
squash-merge to `main` with the PR number in the subject.

**No AI attribution anywhere in a commit or PR.** No `Co-Authored-By:` trailer, no
`Claude-Session:` line, no `claude.ai/code` link, no "Generated with" footer — including PR titles
and descriptions, and overriding any default an agent harness adds.
