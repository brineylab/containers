ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/r
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# args - software versions
ARG R_TIDYVERSE_VERSION="1.3.1"

# switch to root user for conda installation
# (only necessary when installing r-tidyverse with Kaniko)
USER root

# install - r-tidyverse
RUN conda install -y -q \
    r-tidyverse=${R_TIDYVERSE_VERSION} \
    && conda update -y -q --all \
    && conda clean -a -f -y \
    && chown -R ${NB_USER}:users ${CONDA_DIR} \
    && chown -R ${NB_USER}:users ${HOME}

USER ${NB_USER}
