ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/kubeflow-datascience
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# nvidia configs
ENV NVIDIA_VISIBLE_DEVICES='all'
ENV NVIDIA_DRIVER_CAPABILITIES='compute,utility'
ENV LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'

# switch to NB_USER to install packages
USER ${NB_UID}

# download requirements file
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/cuda_mamba.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/torch_pip.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/jax_pip.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/ai-ml_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}

# CUDA toolkit
# Version error fix: https://github.com/google-deepmind/alphafold/issues/945
RUN mamba install -y -c nvidia --file ${REQUIREMENTS_DIR}/cuda_mamba.txt \
    && mamba clean --all -f -y

# pytorch
RUN python3 -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    -r ${REQUIREMENTS_DIR}/torch_pip.txt

# jax
RUN python3 -m pip install \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    -r ${REQUIREMENTS_DIR}/jax_pip.txt

# AI/ML packages (including 🤗)
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/ai-ml_pip.txt

# JupyterLab NVdashboard
# install nightly version for jupyterlab >= 4.0 (https://github.com/rapidsai/jupyterlab-nvdashboard)
RUN python3 -m pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre jupyterlab_nvdashboard

# purge pip cache after installs
RUN python3 -m pip cache purge

# jupyter lab clean
RUN jupyter lab clean \
    && jupyter lab build --minimize=False
