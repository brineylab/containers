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

# models directory
ENV MODELS_DIR="/models"
RUN mkdir "${MODELS_DIR}" \
    && chown "${NB_USER}" "${MODELS_DIR}" \
    && fix-permissions "${MODELS_DIR}"

# switch to NB_USER to install packages
USER ${NB_UID}

# download requirements file
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/cuda_mamba.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/torch_pip.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/jax_pip.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/ai-ml_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}

# CUDA toolkit and cuDNN
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
#     && sudo dpkg -i cuda-keyring_1.1-1_all.deb \
#     && rm cuda-keyring_1.1-1_all.deb \
#     && sudo apt-get update \
#     && sudo apt-get -y install cuda-toolkit-12-6 cudnn

# Version error fix: https://github.com/google-deepmind/alphafold/issues/945
# RUN mamba install -y -c nvidia --file ${REQUIREMENTS_DIR}/cuda_mamba.txt \

RUN mamba install --yes -c nvidia "cuda-toolkit~=12.4" \
    && mamba clean --all -f -y \
    # link cuda directory (installed by mamba) to /usr/local/cuda
    # without this, jax can't find libdevice.10.bc 
    # see: https://github.com/jax-ml/jax/issues/4452
    && sudo mkdir -p /usr/local/cuda \
    && sudo ln -s /opt/conda/* /usr/local/cuda/


RUN mamba install --yes -c nvidia \
    "cuda~=12.4" \
    "cuda-tools~=12.4" \
    # "cuda-toolkit~=12.4" \
    "cuda-version~=12.4" \
    "cuda-command-line-tools~=12.4" \
    "cuda-compiler~=12.4" \
    "cuda-runtime~=12.4" \
    && mamba clean --all -f -y 
    # # link cuda directory (installed by mamba) to /usr/local/cuda
    # # without this, jax can't find libdevice.10.bc 
    # # see: https://github.com/jax-ml/jax/issues/4452
    # && sudo mkdir -p /usr/local/cuda \
    # && sudo ln -s /opt/conda/* /usr/local/cuda/
    

# RUN python3 -m pip install nvidia-pyindex
# RUN python3 -m pip install \
#     nvidia-cuda-runtime-cu12 \
#     nvidia-cuda-cccl-cu12 \
#     nvidia-cuda-cupti-cu12 \
#     nvidia-cuda-nvcc-cu12 \
#     nvidia-cuda-opencl-cu12 \
#     nvidia-cuda-nvrtc-cu12 \
#     nvidia-cublas-cu12 \
#     nvidia-cuda-sanitizer-api-cu12 \
#     nvidia-cufft-cu12 \
#     nvidia-curand-cu12 \
#     nvidia-cusolver-cu12 \
#     nvidia-cusparse-cu12 \
#     nvidia-npp-cu12 \
#     nvidia-nvfatbin-cu12 \
#     nvidia-nvjitlink-cu12 \
#     nvidia-nvjpeg-cu12 \
#     nvidia-nvml-dev-cu12 \
#     nvidia-nvtx-cu12


# pytorch
RUN python3 -m pip install torch
    # --index-url https://download.pytorch.org/whl/cu124
    # --find-links https://download.pytorch.org/whl/torch_stable.html \
    # -r ${REQUIREMENTS_DIR}/torch_pip.txt

# jax
RUN python3 -m pip install "jax[cuda12_local]" 

    # "jax[cuda12_local]"
    # --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    # -r ${REQUIREMENTS_DIR}/jax_pip.txt

# # tensorflow
# RUN python3 -m pip install "tensorflow[and-cuda]"

# AI/ML packages (including 🤗)
RUN python3 -m pip install -r "${REQUIREMENTS_DIR}/ai-ml_pip.txt"

# JupyterLab NVdashboard
# install nightly version for jupyterlab >= 4.0 (https://github.com/rapidsai/jupyterlab-nvdashboard)
RUN python3 -m pip install \
    --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    --pre \
    jupyterlab_nvdashboard


#-------------------
#    ColabDesign
#-------------------

# aria2c
RUN sudo apt-get update \
    && sudo apt-get install --yes aria2 \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# RFdiffusion
RUN cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/RosettaCommons/RFdiffusion.git" \
    && cd RFdiffusion \
    # download model weights
    && mkdir models \
    && cd models \
    && echo "downloading RFdiffusion model weights:" \
    && echo "  - Base_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt \
    && echo "  - Complex_base_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt \
    && echo "  - Complex_Fold_base_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt \
    && echo "  - InpaintSeq_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt \
    && echo "  - InpaintSeq_Fold_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt \
    && echo "  - ActiveSite_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt \
    && echo "  - Base_epoch8_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt \
    && echo "  - Complex_beta_ckpt.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt \
    && echo "  - RF_structure_prediction_weights.pt" \
    && wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt \
    # dependencies
    && python3 -m pip install jedi omegaconf hydra-core icecream pyrsistent pynvml decorator \
    && python3 -m pip install "git+https://github.com/NVIDIA/dllogger#egg=dllogger" \
    && python3 -m pip install --no-dependencies dgl==2.0.0 -f "https://data.dgl.ai/wheels/cu121/repo.html" \
    && python3 -m pip install --no-dependencies e3nn==0.3.3 opt_einsum_fx \
    # install SE(3)-Transformers
    && cd ../env/SE3Transformer \ 
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install -e . \
    # install RFdiffusion
    && cd ../.. \
    && python3 -m pip install -e . \
    && fix-permissions "${NGS_TOOLS_DIR}/RFdiffusion"
ENV DGLBACKEND="pytorch"

# AnAnaS -> https://team.inria.fr/nano-d/software/ananas/
RUN cd "${NGS_TOOLS_DIR}" \
    && wget -qnc "https://files.ipd.uw.edu/krypton/ananas" \
    && sudo chmod +x ananas \
    && sudo cp ananas /usr/local/bin/
    # && fix-permissions "${NGS_TOOLS_DIR}"

# AlphaFold weights (for ColabDesign)
ENV ALPHAFOLD2_PARAMS_DIR="/models/alphafold2/params"
RUN mkdir -p "${ALPHAFOLD2_PARAMS_DIR}" \
    && chown "${NB_USER}" "${ALPHAFOLD2_PARAMS_DIR}" \
    && fix-permissions "${ALPHAFOLD2_PARAMS_DIR}"
RUN cd "${ALPHAFOLD2_PARAMS_DIR}" \
    && aria2c -q -x 16 "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
    && tar -xvf "alphafold_params_2022-12-06.tar" \
    && rm "alphafold_params_2022-12-06.tar" \
    && fix-permissions "${ALPHAFOLD2_PARAMS_DIR}"

# install ColabDesign
RUN cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/sokrypton/ColabDesign" \
    && cd ColabDesign \
    && python3 -m pip install . \
    && fix-permissions "${NGS_TOOLS_DIR}/ColabDesign"

# Rosetta (and PyRosetta)
ENV CXX=/usr/bin/g++
ENV CC=/usr/bin/gcc
RUN cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/RosettaCommons/Rosetta.git" \
    && cd Rosetta/source/ \
    && ./scons.py -j12 mode=release bin \
    && cd src/python/PyRosetta \
    && python3 build.py -j12 \
    && fix-permissions "${NGS_TOOLS_DIR}/Rosetta"


#-------------------------
#    Ab/protein models
#-------------------------

RUN python3 -m pip install \
    # protein models
    boltz \
    fair-esm \
    # chai_lab \
    # antibody models
    ablang2 \
    # igfold \
    ImmuneBuilder \
    "git+https://github.com/OliverT1/p-IgGen.git" 

# RUN cd "${NGS_TOOLS_DIR}" \
#     && git clone "https://github.com/oxpig/AntiFold" \
#     && cd AntiFold \
#     && python3 -m pip install . \
#     && fix-permissions "${NGS_TOOLS_DIR}/AntiFold"


# purge pip cache after installs
RUN python3 -m pip cache purge

# # jupyter lab clean
# RUN jupyter lab clean \
#     && jupyter lab build --minimize=False
