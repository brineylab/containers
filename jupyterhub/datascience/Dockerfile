ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/jupyterhub-jupyter
FROM $BASE_IMG

# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified and maintained by the Briney Lab.
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed 
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license

# OpenJDK version
ARG openjdk_version="17"
ENV OPENJDK_VERSION="${openjdk_version}"


# ----------------------------
#       OS dependencies
# ----------------------------
USER root
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
        # R pre-requisites
        fonts-dejavu \
        gfortran \
        gcc \
        # for PyRosetta
        # cmake \
        # ninja-build \
        # g++ \
        # clang \
        # python3.11-dev \
        # for NGS tools
        "openjdk-${OPENJDK_VERSION}-jre-headless" \
        ca-certificates-java \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# macOS Rosetta virtualization creates junk directory which gets owned by root further up.
# It'll get re-created, but as USER runner after the next directive so hopefully should not cause permission issues.
# More info: https://github.com/jupyter/docker-stacks/issues/2296
RUN rm -rf "/home/${NB_USER}/.cache/"

#-----------------
#      SciPy
#-----------------
USER ${NB_UID}
WORKDIR /tmp
COPY --chown=${NB_UID}:${NB_GID} ./requirements/scipy_mamba.txt ${REQUIREMENTS_DIR}/scipy_mamba.txt
RUN mamba install --yes --file "${REQUIREMENTS_DIR}/scipy_mamba.txt" \
    && mamba clean --all -f -y \
    && npm cache clean --force \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


#-----------------------------
#   ab[x] and dependencies
#-----------------------------
COPY --chown=${NB_UID}:${NB_GID} ./requirements/abx_pip.txt ${REQUIREMENTS_DIR}/abx_pip.txt
RUN python3 -m pip install --no-cache-dir -r ${REQUIREMENTS_DIR}/abx_pip.txt

# ab[x]
# installed separately to prevent pip resolver issues
RUN python3 -m pip install --no-cache-dir \
        "scab>=0.2.4" \
        abstar \
        abutils


#-----------------
#       R
#-----------------
# R packages including IRKernel which gets installed globally.
COPY --chown=${NB_UID}:${NB_GID} ./requirements/r_mamba.txt ${REQUIREMENTS_DIR}/r_mamba.txt
RUN mamba install --yes --file ${REQUIREMENTS_DIR}/r_mamba.txt \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# Install Bioconductor packages (BiocManager installed by mamba above)
RUN Rscript -e "BiocManager::install(c('biomaRt', 'DESeq2'), ask = FALSE, update = FALSE)"

# install R packages that are not available in mamba with CRAN
# multi-thread compilation to speed up duckdb install - use 4 cores (25% of cores with xlarge runner)
COPY --chown=${NB_UID}:${NB_GID} ./requirements/r_cran.txt ${REQUIREMENTS_DIR}/r_cran.txt
ENV MAKEFLAGS="-j4"
RUN Rscript -e "\
  pkgs <- scan('${REQUIREMENTS_DIR}/r_cran.txt', what = '', quiet = TRUE); \
  install.packages( \
    pkgs, \
    dependencies = TRUE, \
    repos = 'https://cran.rstudio.com' \
  )"


#-------------------
#     NGS TOOLS
#-------------------
# cutadapt
RUN python3 -m pip install \
    "cutadapt" \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}" \
    && python3 -m pip cache purge

# NGS directory
USER root
ENV NGS_TOOLS_DIR="/tools"
RUN mkdir "${NGS_TOOLS_DIR}" \
    && chown "${NB_USER}" "${NGS_TOOLS_DIR}" \
    && fix-permissions "${NGS_TOOLS_DIR}"
WORKDIR "${NGS_TOOLS_DIR}"

# bcl2fastq
RUN wget -q "https://burtonlab.s3.amazonaws.com/software/bcl2fastq2_0v2.20.0.422-2_amd64.deb" \ 
    && dpkg -i "bcl2fastq2_0v2.20.0.422-2_amd64.deb" \
    && rm "bcl2fastq2_0v2.20.0.422-2_amd64.deb" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# bases2fastq
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/bases2fastq-1.4.0.tar.gz" \
    && tar xzvf "bases2fastq-1.4.0.tar.gz" \
    && rm "bases2fastq-1.4.0.tar.gz" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# FastQC
RUN wget -q "http://burtonlab.s3.amazonaws.com/software/fastqc_v0.11.9.zip" \
    && unzip "fastqc_v0.11.9.zip" \
    && rm "fastqc_v0.11.9.zip" \
    && chmod +x "${NGS_TOOLS_DIR}/FastQC/fastqc" \
    && ln -s "${NGS_TOOLS_DIR}/FastQC/fastqc" "/usr/local/bin/fastqc" \
    && fix-permissions "/home/${NB_USER}"

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
    && ldconfig

# sickle
# requires zlib
RUN git clone -b v1.2.11 "https://github.com/madler/zlib/" \
    && cd zlib \
    && ./configure \
    && make \
    && make install \
    && cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/najoshi/sickle" \
    && cd sickle \
    && make \
    && ln -s "${NGS_TOOLS_DIR}/sickle/sickle" "/usr/local/bin/sickle" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# CellRanger
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/cellranger-8.0.1.tar" \
    && tar xf "cellranger-8.0.1.tar" \
    && rm "cellranger-8.0.1.tar" \
    && sudo chmod -R 777 "${NGS_TOOLS_DIR}/cellranger-8.0.1" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}" 

# IgDiscover
RUN git clone "https://gitlab.com/gkhlab/igdiscover22.git" \
    && cd igdiscover22 \
    && sed '/- defaults/d' environment.yml > environment_no-default.yml \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# macOS Rosetta virtualization creates junk directory which gets owned by root further up.
# It'll get re-created, but as USER runner after the next directive so hopefully should not cause permission issues.
# More info: https://github.com/jupyter/docker-stacks/issues/2296
RUN rm -rf "/home/${NB_USER}/.cache/"

USER ${NB_UID}
RUN cd "${NGS_TOOLS_DIR}/igdiscover22" \
    && mamba create -f environment_no-default.yml -n igdiscover \
    && mamba run -n igdiscover python3 -m pip install --no-deps -e . \
    && mamba run -n igdiscover python3 -m pip install "networkx==2.1" \
    && mamba clean --all -f -y

# dorado (ONT)
RUN cd "${NGS_TOOLS_DIR}" \
    && wget -q "https://brineylab.s3.amazonaws.com/tools/dorado-0.7.0-linux-x64.tar.gz" \
    && tar xzf "dorado-0.7.0-linux-x64.tar.gz" \
    && rm "dorado-0.7.0-linux-x64.tar.gz" \
    && sudo chmod -R 777 "${NGS_TOOLS_DIR}/dorado-0.7.0-linux-x64" \
    && fix-permissions "${NGS_TOOLS_DIR}/dorado-0.7.0-linux-x64" \
    && fix-permissions "/home/${NB_USER}"

# update path with all tools
ENV PATH="${NGS_TOOLS_DIR}:${NGS_TOOLS_DIR}/cellranger-8.0.1/bin:${NGS_TOOLS_DIR}/dorado-0.7.0-linux-x64/bin:$PATH"

# ----------------------------
#         Final Setup
# ----------------------------
# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR "${HOME}"
