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
    cmake \
    ninja-build \
    g++ \
    clang \
    python3.11-dev \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*


#----------------------
#     CODESERVER
#----------------------
USER root

# install - code-server
RUN curl -sL "https://github.com/cdr/code-server/releases/download/${CODESERVER_VERSION}/code-server_${CODESERVER_VERSION/v/}_amd64.deb" -o /tmp/code-server.deb \
    && dpkg -i /tmp/code-server.deb \
    && rm -f /tmp/code-server.deb

USER ${NB_UID}

# code-server extensions
RUN code-server --install-extension ms-python.black-formatter --force \
    && code-server --install-extension ms-python.python --force \
    && code-server --install-extension ms-toolsai.jupyter --force \
    && code-server --install-extension redhat.vscode-yaml --force \
    && code-server --install-extension redhat.vscode-xml --force \
    && code-server --install-extension zhuangtongfa.material-theme --force \
    && code-server --list-extensions --show-versions

USER root
RUN mamba install --yes \
    "jupyter-server-proxy" \
    "jupyter-vscode-proxy" \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# set the home directory
ENV CODE_WORKINGDIR="${HOME}/codeserver"


#-----------------------------
#   ab[x] and dependencies
#-----------------------------

COPY --chown=${NB_UID}:${NB_GID} ./requirements/abx_pip.txt ${REQUIREMENTS_DIR}/abx_pip.txt
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/abx_pip.txt \
    && python3 -m pip cache purge


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
# multi-thread compilation to speed up duckdb install
COPY --chown=${NB_UID}:${NB_GID} ./requirements/r_cran.txt ${REQUIREMENTS_DIR}/r_cran.txt
ENV MAKEFLAGS="-j$(nproc)"
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
ENV NGS_TOOLS_DIR="/tools"
RUN mkdir "${NGS_TOOLS_DIR}" \
    && chown "${NB_USER}" "${NGS_TOOLS_DIR}" \
    && fix-permissions "${NGS_TOOLS_DIR}"
WORKDIR "${NGS_TOOLS_DIR}"

# bcl2fastq
# RUN wget -q "https://burtonlab.s3.amazonaws.com/software/bcl2fastq2_0v2.20.0.422-2_amd64.deb" \ 
#     && dpkg -i "bcl2fastq2_0v2.20.0.422-2_amd64.deb" \
#     && rm "bases2fastq-1.4.0.tar.gz" \
#     && fix-permissions "${NGS_TOOLS_DIR}" \
#     && fix-permissions "/home/${NB_USER}"

# bases2fastq
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/bases2fastq-1.4.0.tar.gz" \
    && tar xzvf "bases2fastq-1.4.0.tar.gz" \
    && rm "bases2fastq-1.4.0.tar.gz" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# FastQC
# RUN wget -q "http://burtonlab.s3.amazonaws.com/software/fastqc_v0.11.9.zip" \
#     && unzip "fastqc_v0.11.9.zip" \
#     && rm "fastqc_v0.11.9.zip" \
#     && chmod +x "${NGS_TOOLS_DIR}/FastQC/fastqc" \
#     && ln -s "${NGS_TOOLS_DIR}/FastQC/fastqc" "/usr/local/bin/fastqc" \
#     && fix-permissions "/home/${NB_USER}"

# PANDAseq
RUN apt-get update \
    && apt-get install -y \
        "build-essential" \
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
# RUN wget -q "http://burtonlab.s3.amazonaws.com/software/zlib1211.zip" \
#     && unzip "zlib1211.zip" \ 
#     && rm "zlib1211.zip"
#     && cd "${NGS_TOOLS_DIR}/zlib-1.2.11" \
#     && ./configure \
#     && make \
#     && make install \
#     && cd "${NGS_TOOLS_DIR}" \
#     && git clone "https://github.com/najoshi/sickle" \
#     && cd sickle \
#     && make \
#     && ln -s "${NGS_TOOLS_DIR}/sickle/sickle" "/usr/local/bin/sickle" \
#     && fix-permissions "${NGS_TOOLS_DIR}" \
#     && fix-permissions "/home/${NB_USER}"

# CellRanger
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/cellranger-8.0.1.tar" \
    && tar xf "cellranger-8.0.1.tar" \
    && rm "cellranger-8.0.1.tar" \
    && chmod -R 777 "${NGS_TOOLS_DIR}/cellranger-8.0.1" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}" 

# IgDiscover
RUN git clone "https://gitlab.com/gkhlab/igdiscover22.git" \
    && cd igdiscover22 \
    && mamba install --yes \
        nomkl \
        "snakemake-minimal>=4.5" \
        scipy \
        "muscle=3.8.1551" \
        pear \
        flash \
        "igblast=1.19" \
        "ruamel.yaml>=0.15" \
        "vsearch=2.15.2" \
        dnaio \
        "xopen>=1.2.0" \
        "tinyalign>=0.2" \
        regex \
        "networkx=2.5" \
        "minimap2=2.28" \
    && python3 -m pip install --no-deps -e . \
    && python3 -m pip install "networkx==2.1" \
    && mamba clean --all -f -y \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"


# dorado (ONT)
RUN cd "${NGS_TOOLS_DIR}" \
    && wget -q "https://brineylab.s3.amazonaws.com/tools/dorado-0.7.0-linux-x64.tar.gz" \
    && tar xzf "dorado-0.7.0-linux-x64.tar.gz" \
    && rm "dorado-0.7.0-linux-x64.tar.gz" \
    && chmod -R 777 "${NGS_TOOLS_DIR}/dorado-0.7.0-linux-x64" \
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
