# Images were initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and were subsequently modified by the Briney lab.

ARG BASE_IMG=kubeflownotebookswg/codeserver-python
FROM $BASE_IMG

USER root

ENV REQUIREMENTS_DIR="${HOME}/.requirements"
RUN mkdir -p ${REQUIREMENTS_DIR} \
    && chown -R ${NB_USER}:users ${REQUIREMENTS_DIR}

RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc \
    g++ \
    sudo \
    wget \
    htop \ 
    btop \
    tmux \
    screen \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# download requirements files
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/cuda_mamba.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/scipy_mamba.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/abx_pip.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/torch_pip.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/jax_pip.txt \
    && wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/ai-ml_pip.txt \
    && chown -R ${NB_USER}:users ${REQUIREMENTS_DIR}

USER $NB_UID

# CUDA toolkit
# Version error fix: https://github.com/google-deepmind/alphafold/issues/945
RUN mamba install -y -c nvidia --file ${REQUIREMENTS_DIR}/cuda_mamba.txt \
    && mamba clean --all -f -y

# scipy
RUN mamba install --yes --file ${REQUIREMENTS_DIR}/scipy_mamba.txt \
    && mamba clean --all -f -y

# fix ownership
USER root
RUN chown -R ${NB_USER}:users "${CONDA_DIR}" \
    && chown -R ${NB_USER}:users "/home/${NB_USER}"
USER $NB_UID

# ab[x] and dependencies
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/abx_pip.txt

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

# purge pip cache after installs
RUN python3 -m pip cache purge

# code-server extensions
RUN code-server --install-extension ms-python.black-formatter --force \
    && code-server --install-extension ms-python.python --force \
    && code-server --install-extension ms-toolsai.jupyter --force \
    && code-server --install-extension redhat.vscode-yaml --force \
    && code-server --install-extension redhat.vscode-xml --force \
    && code-server --install-extension zhuangtongfa.material-theme --force \
    && code-server --list-extensions --show-versions

# s6 - 01-copy-tmp-home
USER root
RUN mkdir -p /tmp_home \
    && cp -r ${HOME} /tmp_home \
    && chown -R ${NB_USER}:users /tmp_home
USER $NB_UID