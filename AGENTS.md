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
| Build and publish | `.github/workflows/docker-publish.yml` |

## How builds happen

**Images publish on a GitHub release**, not on merge — tagged with both the release tag and
`latest`. A merged PR changes nothing that anyone is running until a release is cut. Manual runs
are possible via `workflow_dispatch`, which also exposes `use_cache` and `push` inputs.

The build graph is expressed in the workflow's `needs:`, so `base` → `jupyterhub-base`,
`datascience` → `jupyterhub-datascience`, `deeplearning` → `jupyterhub-deeplearning`. Builds run
with the **repository root** as the Docker context, so `COPY` paths are written relative to the
root, not to the Dockerfile.

## Tests

`pytest` against built images, `tests/` — one file per image plus shared checks in
`test_base_images.py`. The `gpu` marker means "requires a GPU on the host" and is skipped
otherwise, so a green run on a CPU box has not exercised the CUDA paths. Check which markers
actually ran before concluding an image is good.

## Conventions

**Add packages in `requirements/`, not in a Dockerfile.** The split is by package manager and
audience (`apt.txt`, `pip.txt`, `ai-ml_pip.txt`, `r_conda.txt`, `r_cran.txt`, `jupyter_pip.txt`),
and the `jupyterhub-*` images inherit through their parents — so a line added to `apt.txt` reaches
all six images. That inheritance is the reason to resist adding a package "just for one image".

**Pin what you can reason about.** These images are the reproducibility floor for lab analyses;
an unpinned dependency changes what a rebuilt image does without any commit here.

**Commit messages use Conventional Commits** — `type(scope): subject`, types `feat`/`fix`/`docs`/
`refactor`/`test`/`chore`, scope usually the image or `requirements`. Work on a branch and
squash-merge to `main` with the PR number in the subject.

**No AI attribution anywhere in a commit or PR.** No `Co-Authored-By:` trailer, no
`Claude-Session:` line, no `claude.ai/code` link, no "Generated with" footer — including PR titles
and descriptions, and overriding any default an agent harness adds.
