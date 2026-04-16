# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified and maintained by the Briney Lab.

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license

# NVIDIA CUDA DL Base provides CUDA, cuDNN, NCCL, and TensorRT without Python.
# PyTorch and other frameworks are installed via pip.
ARG BASE_IMG=nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04
FROM $BASE_IMG

LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# Python version
ARG PYTHON_VERSION=3.12

# Environmental variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# NVIDIA configs
ENV NVIDIA_VISIBLE_DEVICES='all'
ENV NVIDIA_DRIVER_CAPABILITIES='compute,utility'

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]


# ----------------------------
#       OS dependencies
# ----------------------------
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
#         Micromamba
# ----------------------------
COPY ./common/initial-condarc "${CONDA_DIR}/.condarc"
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
        'conda' \
        'mamba' \
    && rm -rf /tmp/bin/ \
    && mamba list --full-name 'python' | awk 'END{sub("[^.]*$", "*", $2); print $1 " " $2}' >> "${CONDA_DIR}/conda-meta/pinned" \
    && mamba clean --all -f -y \
    # move conda compiler_compat/ld to force use of system linker (fixes deepspeed JIT compilation)
    # ref: https://github.com/ContinuumIO/anaconda-issues/issues/11152
    && mv "${CONDA_DIR}/compiler_compat/ld" "${CONDA_DIR}/compiler_compat/ld.bak"

# Mamba shell hook for interactive sessions, with base activation
# Set umask to 000
RUN echo 'eval "$(mamba shell hook --shell bash)"' >> /etc/bash.bashrc \
    && echo 'mamba activate base' >> /etc/bash.bashrc \
    && echo 'umask 000' >> /etc/bash.bashrc


# ----------------------------
#      R Packages (conda)
# ----------------------------
COPY ./requirements/r_conda.txt /tmp/r_conda.txt
RUN mamba install --yes --file /tmp/r_conda.txt \
    && mamba clean --all -f -y


# ----------------------------
#      Pip Packages
# ----------------------------
RUN python3 -m pip install --no-cache-dir uv

COPY ./requirements/pip.txt /tmp/pip.txt
RUN uv pip install --no-cache --system -r /tmp/pip.txt


# ----------------------------
#      AI/ML Packages
# ----------------------------
COPY ./requirements/ai-ml_pip.txt /tmp/ai-ml_pip.txt
RUN uv pip install --no-cache --system -r /tmp/ai-ml_pip.txt


# ----------------------------
#    R CRAN / Bioconductor
# ----------------------------
# Bioconductor packages (BiocManager installed by conda above)
RUN Rscript -e "BiocManager::install(c('biomaRt', 'DESeq2'), ask = FALSE, update = FALSE)"

# CRAN packages
COPY ./requirements/r_cran.txt /tmp/r_cran.txt
RUN MAKEFLAGS="-j$(nproc)" Rscript -e "\
  pkgs <- scan('/tmp/r_cran.txt', what = '', quiet = TRUE); \
  install.packages( \
    pkgs, \
    dependencies = TRUE, \
    repos = 'https://cran.rstudio.com' \
  )"


# ----------------------------
#         Final Setup
# ----------------------------
ENV HOME=/home/jovyan
RUN mkdir -p /home/jovyan
WORKDIR /home/jovyan

ENTRYPOINT ["tini", "--"]
CMD ["/bin/bash"]
