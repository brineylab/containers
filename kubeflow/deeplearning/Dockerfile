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

# switch to NB_USER to install python packages
USER ${NB_UID}

# CUDA toolkit
RUN conda update --force conda
RUN conda install --yes cuda -c nvidia \
    && conda clean --all -f -y

# pytorch
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/torch_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    -r ${REQUIREMENTS_DIR}/torch_pip.txt
RUN python3 -m pip cache purge

# jax
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/jax_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    -r ${REQUIREMENTS_DIR}/jax_pip.txt
RUN python3 -m pip cache purge

# AI/ML packages (including 🤗)
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/ai-ml_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/ai-ml_pip.txt
RUN python3 -m pip cache purge

# jupyterlab extensions
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/nvidia_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/nvidia_pip.txt
RUN python3 -m pip cache purge

# jupyter lab clean
RUN jupyter lab clean \
    && jupyter lab build --minimize=False
