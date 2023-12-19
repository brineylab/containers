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

# switch to NB_USER to install python packages
USER ${NB_UID}

# pytorch
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/torch_cpu_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    -r ${REQUIREMENTS_DIR}/torch_pip.txt

# jax
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/jax_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    -r ${REQUIREMENTS_DIR}/jax_pip.txt

# AI/ML packages (including 🤗)
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/ai-ml_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/ai-ml_pip.txt

# jupyterlab extensions
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/nvidia_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/nvidia_pip.txt

# jupyter lab clean
RUN jupyter lab clean \
    && jupyter lab build --minimize=False
