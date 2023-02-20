ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/datascience
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

# pytorch
RUN python3 -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==1.13.0+cu117 \
    torchvision==0.14.0+cu117


# # tensorflow
# RUN mamba install --yes \
#     -c conda-forge \
#     cudatoolkit=11.2 \
#     cudnn=8.1.0 && \
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ && \
#     python3 -m pip install tensorflow && \
#     # verify install:
#     python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# hugging face
RUN python3 -m pip install \
    transformers \
    diffusers \
    datasets


# jupyterlab extensions
RUN python3 -m pip install \
    jupyterlab_nvdashboard \
    nvitop
