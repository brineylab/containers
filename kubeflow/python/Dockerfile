ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/kubeflow-base
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# set shell to bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# switch to NB_UID for installs
USER ${NB_UID}

WORKDIR /tmp

# kubeflow
RUN mamba install --yes \
   'kfp' \
   'kfp-server-api' \
   && mamba clean --all -f -y \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}"

# scipy
RUN wget --directory-prefix="${REQUIREMENTS_DIR}" https://raw.githubusercontent.com/briney/containers/main/requirements/scipy_mamba.txt \
   && fix-permissions ${REQUIREMENTS_DIR}
RUN mamba install --yes --file ${REQUIREMENTS_DIR}/scipy_mamba.txt \
   && mamba clean --all -f -y \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}"

# kfserving
RUN python3 -m pip install kfserving

# need to be root again to do some s6 stuff
USER root

# s6 - copy scripts
COPY s6/ /etc
RUN chown -R ${NB_USER}:users /etc/s6
RUN chmod -R 755 /etc/s6

# s6 - 01-copy-tmp-home
RUN mkdir -p /tmp_home \
   && cp -r ${HOME} /tmp_home \
   && chown -R ${NB_USER}:users /tmp_home

USER ${NB_UID}
