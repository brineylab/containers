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
RUN python3 -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==1.13.0+cu117 \
    torchvision==0.14.0+cu117

# hugging face
RUN python3 -m pip install \
    transformers \
    diffusers \
    datasets

# jupyterlab extensions
RUN python3 -m pip install \
    jupyterlab_nvdashboard \
    nvitop