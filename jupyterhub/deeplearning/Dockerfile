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


# ----------------------------
#         CUDA Setup
# ----------------------------
USER root

# NVIDIA configs
ENV NVIDIA_VISIBLE_DEVICES='all'
ENV NVIDIA_DRIVER_CAPABILITIES='compute,utility'
ENV LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'

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
    && mkdir -p /usr/local/cuda \
    && ln -s /opt/conda/* /usr/local/cuda/


# ----------------------------
#         AI/ML Packages
# ----------------------------
COPY --chown=${NB_UID}:${NB_GID} ./requirements/ai-ml_pip.txt ${REQUIREMENTS_DIR}/ai-ml_pip.txt
RUN python3 -m pip install -r "${REQUIREMENTS_DIR}/ai-ml_pip.txt" \
    # JupterLab NVdashboard (nightly for jupyterlab >= 4.0)
    # https://github.com/rapidsai/jupyterlab-nvdashboard
    && python3 -m pip install \
        --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
        --pre \
        jupyterlab_nvdashboard \
    && python3 -m pip cache purge


# ----------------------------
#         Final Setup
# ----------------------------
# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR "${HOME}"
