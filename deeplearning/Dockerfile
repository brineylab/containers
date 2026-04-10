# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified and maintained by the Briney Lab.

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license

# The NVIDIA PyTorch base image provides CUDA, cuDNN, NCCL, and Python 3.12.
# No manual CUDA installation is needed.
ARG BASE_IMG=nvcr.io/nvidia/pytorch:26.03-py3
FROM $BASE_IMG

ARG OWNER=brineylab
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# NB user
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Python version (must match the NVIDIA base image's Python)
ARG PYTHON_VERSION=3.12

# Environmental variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=${NB_USER} \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV HOME="/home/${NB_USER}"
ENV PATH="${HOME}/.local/bin:${PATH}:${CONDA_DIR}/bin"
ENV PYTHONPATH="${CONDA_DIR}/lib/python${PYTHON_VERSION}/site-packages"
ENV REQUIREMENTS_DIR="${HOME}/.requirements"

# NVIDIA configs
ENV NVIDIA_VISIBLE_DEVICES='all'
ENV NVIDIA_DRIVER_CAPABILITIES='compute,utility'

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]


# ----------------------------
#       OS dependencies
# ----------------------------
USER root
COPY ./requirements/apt.txt /tmp/apt.txt
RUN apt-get update --yes \
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends \
        $(grep -v '^\s*#' /tmp/apt.txt | grep -v '^\s*$') \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen


# ----------------------------
#       User Setup
# ----------------------------
COPY --chown=${NB_UID}:${NB_GID} ./common/fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
    echo 'eval "$(conda shell.bash hook)"' >> /etc/skel/.bashrc

# Delete existing user with UID="${NB_UID}" if it exists
# hadolint ignore=SC2046
RUN if grep -q "${NB_UID}" /etc/passwd; then \
        userdel --remove $(id -un "${NB_UID}"); \
    fi

# Create NB_USER with name jovyan user with UID=1000 and in the 'users' group
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su \
    && sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers \
    && sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers \
    && useradd -l -m -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" \
    && mkdir -p "${CONDA_DIR}" \
    && chown "${NB_USER}:${NB_GID}" "${CONDA_DIR}" \
    && chmod g+w /etc/passwd \
    && fix-permissions "${HOME}" \
    && fix-permissions "${CONDA_DIR}" \
    && usermod -aG sudo ${NB_USER} \
    && echo "${NB_USER}  ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/${NB_USER}


# ----------------------------
#         Micromamba
# ----------------------------
# Install micromamba alongside NVIDIA base's Python.
# Micromamba is used for R and conda-only packages.
USER ${NB_UID}
COPY --chown="${NB_UID}:${NB_GID}" ./common/initial-condarc "${CONDA_DIR}/.condarc"
WORKDIR /tmp
RUN set -x \
    && arch=$(uname -m) \
    && if [ "${arch}" = "x86_64" ]; then \
        arch="64"; \
    fi \
    && wget --progress=dot:giga -O - \
        "https://micro.mamba.pm/api/micromamba/linux-${arch}/latest" | tar -xvj bin/micromamba \
    && ./bin/micromamba install \
        --root-prefix="${CONDA_DIR}" \
        --prefix="${CONDA_DIR}" \
        --yes \
        "python=${PYTHON_VERSION}" \
        'jupyter_core' \
        'conda' \
        'mamba' \
    && rm -rf /tmp/bin/ \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


# ----------------------------
#         Directories
# ----------------------------
RUN mkdir -p ${REQUIREMENTS_DIR} \
    && mkdir "/home/${NB_USER}/work"


# ----------------------------
#      Conda Packages
# ----------------------------
COPY --chown=${NB_UID}:${NB_GID} ./requirements/conda.txt ${REQUIREMENTS_DIR}/conda.txt
RUN mamba install --yes --file "${REQUIREMENTS_DIR}/conda.txt" \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}"


# ----------------------------
#      Pip Packages
# ----------------------------
USER root
RUN python3 -m pip install --no-cache-dir --break-system-packages uv

COPY --chown=${NB_UID}:${NB_GID} ./requirements/pip.txt ${REQUIREMENTS_DIR}/pip.txt
COPY --chown=${NB_UID}:${NB_GID} ./requirements/abx_pip.txt ${REQUIREMENTS_DIR}/abx_pip.txt
RUN uv pip install --no-cache --system --break-system-packages -r ${REQUIREMENTS_DIR}/pip.txt -r ${REQUIREMENTS_DIR}/abx_pip.txt

# Rebuild fastcluster against NumPy 2.x (NVIDIA base ships NumPy 2, but the wheel targets NumPy 1)
RUN uv pip install --no-cache --system --break-system-packages --force-reinstall --no-binary fastcluster fastcluster


# ----------------------------
#      AI/ML Packages
# ----------------------------
COPY --chown=${NB_UID}:${NB_GID} ./requirements/ai-ml_pip.txt ${REQUIREMENTS_DIR}/ai-ml_pip.txt
RUN uv pip install --no-cache --system --break-system-packages -r "${REQUIREMENTS_DIR}/ai-ml_pip.txt"


# ----------------------------
#    R CRAN / Bioconductor
# ----------------------------
USER ${NB_UID}
RUN Rscript -e "BiocManager::install(c('biomaRt', 'DESeq2'), ask = FALSE, update = FALSE)" \
    && fix-permissions "${CONDA_DIR}/lib/R"

COPY --chown=${NB_UID}:${NB_GID} ./requirements/r_cran.txt ${REQUIREMENTS_DIR}/r_cran.txt
RUN MAKEFLAGS="-j$(nproc)" Rscript -e "\
  pkgs <- scan('${REQUIREMENTS_DIR}/r_cran.txt', what = '', quiet = TRUE); \
  install.packages( \
    pkgs, \
    dependencies = TRUE, \
    repos = 'https://cran.rstudio.com' \
  )" \
    && fix-permissions "${CONDA_DIR}/lib/R"


# ----------------------------
#         Final Setup
# ----------------------------
USER ${NB_UID}
RUN rm -rf "/home/${NB_USER}/.cache/"
WORKDIR "${HOME}"

ENTRYPOINT ["tini", "--"]
CMD ["/bin/bash"]
