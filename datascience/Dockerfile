# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks/tree/main/images/docker-stacks-foundation
# and was subsequently modified by the Briney Lab.

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license

ARG BASE_IMG=ubuntu:24.04
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
    && mamba clean --all -f -y


# ----------------------------
#      Conda Packages
# ----------------------------
COPY ./requirements/conda.txt /tmp/conda.txt
RUN mamba install --yes --file /tmp/conda.txt \
    && mamba clean --all -f -y


# ----------------------------
#      Pip Packages
# ----------------------------
RUN python3 -m pip install --no-cache-dir uv

COPY ./requirements/pip.txt /tmp/pip.txt
RUN uv pip install --no-cache --system -r /tmp/pip.txt


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
#       NGS dependencies
# ----------------------------
# OpenJDK version (required for NGS tools)
ARG openjdk_version="17"
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
        "openjdk-${openjdk_version}-jre-headless" \
        ca-certificates-java \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


#-------------------
#     NGS TOOLS
#-------------------
ENV NGS_TOOLS_DIR="/tools"
RUN mkdir "${NGS_TOOLS_DIR}"
WORKDIR "${NGS_TOOLS_DIR}"

# bcl2fastq
RUN wget -q "https://burtonlab.s3.amazonaws.com/software/bcl2fastq2_0v2.20.0.422-2_amd64.deb" \
    && dpkg -i "bcl2fastq2_0v2.20.0.422-2_amd64.deb" \
    && rm "bcl2fastq2_0v2.20.0.422-2_amd64.deb"

# bases2fastq
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/bases2fastq-1.4.0.tar.gz" \
    && tar xzvf "bases2fastq-1.4.0.tar.gz" \
    && rm "bases2fastq-1.4.0.tar.gz"

# FastQC
RUN wget -q "http://burtonlab.s3.amazonaws.com/software/fastqc_v0.11.9.zip" \
    && unzip "fastqc_v0.11.9.zip" \
    && rm "fastqc_v0.11.9.zip" \
    && chmod +x "${NGS_TOOLS_DIR}/FastQC/fastqc" \
    && ln -s "${NGS_TOOLS_DIR}/FastQC/fastqc" "/usr/local/bin/fastqc"

# PANDAseq
RUN apt-get update \
    && apt-get install -y \
        "libtool" \
        "automake" \
        "zlib1g-dev" \
        "libbz2-dev" \
        "pkg-config" \
    && git clone "https://github.com/neufeld/pandaseq" \
    && cd pandaseq \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install \
    && ldconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# sickle
RUN git clone -b v1.2.11 "https://github.com/madler/zlib/" \
    && cd zlib \
    && ./configure \
    && make \
    && make install \
    && cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/najoshi/sickle" \
    && cd sickle \
    && make \
    && ln -s "${NGS_TOOLS_DIR}/sickle/sickle" "/usr/local/bin/sickle"

# CellRanger
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/cellranger-8.0.1.tar" \
    && tar xf "cellranger-8.0.1.tar" \
    && rm "cellranger-8.0.1.tar"

# dorado (ONT)
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/dorado-0.7.0-linux-x64.tar.gz" \
    && tar xzf "dorado-0.7.0-linux-x64.tar.gz" \
    && rm "dorado-0.7.0-linux-x64.tar.gz"

# cutadapt
RUN uv pip install --no-cache --system "cutadapt"

# IgDiscover
RUN git clone "https://gitlab.com/gkhlab/igdiscover22.git" \
    && cd igdiscover22 \
    && sed '/- defaults/d' environment.yml > environment_no-default.yml

RUN cd "${NGS_TOOLS_DIR}/igdiscover22" \
    && mamba create -f environment_no-default.yml -n igdiscover \
    && mamba run -n igdiscover python3 -m pip install --no-cache-dir --no-deps -e . \
    && mamba run -n igdiscover python3 -m pip install --no-cache-dir "networkx==2.1" \
    && mamba clean --all -f -y

# update path with all tools
ENV PATH="${NGS_TOOLS_DIR}:${NGS_TOOLS_DIR}/cellranger-8.0.1/bin:${NGS_TOOLS_DIR}/dorado-0.7.0-linux-x64/bin:$PATH"


# ----------------------------
#         Final Setup
# ----------------------------
ENV HOME=/home/jovyan
RUN mkdir -p /home/jovyan
WORKDIR /home/jovyan

ENTRYPOINT ["tini", "--"]
CMD ["/bin/bash"]
