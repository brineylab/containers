ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/jupyterhub-datascience
FROM $BASE_IMG

# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified and maintained by the Briney Lab.
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed 
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license


# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# nvidia configs
ENV NVIDIA_VISIBLE_DEVICES='all'
ENV NVIDIA_DRIVER_CAPABILITIES='compute,utility'
ENV LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'

# switch to NB_USER to install packages
USER ${NB_UID}

# download requirements file
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/cuda_mamba.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/ai-ml_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}

# CUDA toolkit
RUN mamba install --yes -c nvidia \
    "cuda~=12.4" \
    "cuda-tools~=12.4" \
    "cuda-toolkit~=12.4" \
    "cuda-command-line-tools~=12.4" \
    "cuda-compiler~=12.4" \
    "cuda-runtime~=12.4" \
    && mamba clean --all -f -y \
    # link cuda directory (installed by mamba) to /usr/local/cuda
    # without this, jax can't find libdevice.10.bc 
    # see: https://github.com/jax-ml/jax/issues/4452
    && sudo mkdir -p /usr/local/cuda \
    && sudo ln -s /opt/conda/* /usr/local/cuda/

# pytorch
RUN python3 -m pip install torch

# jax
RUN python3 -m pip install "jax[cuda12_local]" 

# AI/ML packages (including 🤗)
RUN python3 -m pip install -r "${REQUIREMENTS_DIR}/ai-ml_pip.txt"

# JupyterLab NVdashboard
# install nightly version for jupyterlab >= 4.0 (https://github.com/rapidsai/jupyterlab-nvdashboard)
RUN python3 -m pip install \
    --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    --pre \
    jupyterlab_nvdashboard

# purge pip cache after installs
RUN python3 -m pip cache purge

