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


# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# R pre-requisites
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    "openjdk-${OPENJDK_VERSION}-jre-headless" \
    ca-certificates-java \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# Configure IPython system-wide
COPY ipython_kernel_config.py "/etc/ipython/"
RUN fix-permissions "/etc/ipython/"

USER ${NB_UID}

# Install pyarrow
RUN mamba install --quiet --yes \
    "pyarrow" \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

WORKDIR "${HOME}"
EXPOSE 4040


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
RUN code-server --install-extension ms-python.black-formatter
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-toolsai.jupyter
RUN code-server --install-extension redhat.vscode-yaml
RUN code-server --install-extension redhat.vscode-xml
RUN code-server --install-extension zhuangtongfa.material-theme

RUN mamba install --yes \
    "jupyter-server-proxy" \
    "jupyter-vscode-proxy" \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# set the home directory
ENV CODE_WORKINGDIR="${HOME}/codeserver"


#----------------------
#      DESKTOP
#----------------------

# USER root

# # install ubuntu desktop GUI
# RUN apt-get update --yes \
#     && apt-get install --yes --no-install-recommends \
#     # "ubuntu-desktop" \
#     "dbus-x11" \ 
#     "libgl1-mesa-glx" \
#     "xorg" \ 
#     "xfce4" \
#     "xfce4-panel" \ 
#     "xfce4-session" \ 
#     "xfce4-settings" \ 
#     "xubuntu-icon-theme" \ 
#     && apt-get remove --yes --quiet light-locker \
#     && apt-get clean \ 
#     && rm -rf /var/lib/apt/lists/*

# USER ${NB_UID}

# # install plugin for accessing the desktop
# # from inside JupyterHub
# RUN python3 -m pip install \
#     "jupyter-remote-desktop-proxy"

# # install websockify
# RUN mamba install --quiet --yes \
#     "websockify" \
#     && mamba clean --all -f -y \
#     && fix-permissions "${CONDA_DIR}" \
#     && fix-permissions "/home/${NB_USER}"


#-----------------------------
#   ab[x] and dependencies
#-----------------------------

# ab[x] and dependencies
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/abx_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/abx_pip.txt
RUN python3 -m pip cache purge


#-----------------
#       R
#-----------------

# R packages including IRKernel which gets installed globally.
# r-e1071: dependency of the caret R package
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/r_mamba.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN mamba install --yes --file ${REQUIREMENTS_DIR}/r_mamba.txt \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


#-----------------
#     Rust
#-----------------

# install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && fix-permissions "${HOME}/.cargo" \
    && fix-permissions "/home/${NB_USER}" \
    && sudo chmod +x "${HOME}/.cargo/env" \
    && "${HOME}/.cargo/env"

ENV PATH="${HOME}/.cargo/bin:$PATH"

# install excvr
RUN cargo install --locked evcxr_jupyter \
    && evcxr_jupyter --install \
    && rustup component add rust-src


#-----------------
#     JULIA
#-----------------

USER root

# Julia installation
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
# Check https://julialang.org/downloads/
# ARG julia_version="1.8.5"  # moved to base

# Julia dependencies
# install Julia packages in /opt/julia instead of ${HOME}
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia
    # JULIA_VERSION="${julia_version}"  # moved to base

WORKDIR /tmp

# hadolint ignore=SC2046
RUN set -x \
    && julia_arch=$(uname -m) \
    && julia_short_arch="${julia_arch}" \
    && if [ "${julia_short_arch}" == "x86_64" ]; then \
    julia_short_arch="x64"; \
    fi; \
    julia_installer="julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" \
    && julia_major_minor=$(echo "${JULIA_VERSION}" | cut -d. -f 1,2) \
    && mkdir "/opt/julia-${JULIA_VERSION}" \
    && wget -q "https://julialang-s3.julialang.org/bin/linux/${julia_short_arch}/${julia_major_minor}/${julia_installer}" \
    && tar xzf "${julia_installer}" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1 \
    && rm "${julia_installer}" \
    && ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

# Show Julia where conda libraries are \
RUN mkdir /etc/julia \
    && echo "push!(Libdl.DL_LOAD_PATH, \"${CONDA_DIR}/lib\")" >> /etc/julia/juliarc.jl \
    # Create JULIA_PKGDIR 
    && mkdir "${JULIA_PKGDIR}" \
    && chown "${NB_USER}" "${JULIA_PKGDIR}" \
    && fix-permissions "${JULIA_PKGDIR}"

USER ${NB_UID}

# Add Julia packages.
# Install IJulia as jovyan and then move the kernelspec out
# to the system share location. Avoids problems with runtime UID change not
# taking effect properly on the .local folder in the jovyan home dir.
RUN julia -e 'import Pkg; Pkg.update()' \
    && julia -e 'import Pkg; Pkg.add("HDF5")' \
    && julia -e 'using Pkg; pkg"add IJulia"; pkg"precompile"' \
    # move kernelspec out of home \
    && mv "${HOME}/.local/share/jupyter/kernels/julia"* "${CONDA_DIR}/share/jupyter/kernels/" \
    && chmod -R go+rx "${CONDA_DIR}/share/jupyter" \
    && rm -rf "${HOME}/.local" \
    && fix-permissions "${JULIA_PKGDIR}" "${CONDA_DIR}/share/jupyter"


#-------------------
#     NGS TOOLS
#-------------------

# cutadapt
RUN python3 -m pip install \
    "cutadapt" \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"
RUN python3 -m pip cache purge


USER root

ENV NGS_TOOLS_DIR="/tools"

RUN mkdir "${NGS_TOOLS_DIR}" \
    && chown "${NB_USER}" "${NGS_TOOLS_DIR}" \
    && fix-permissions "${NGS_TOOLS_DIR}"

ENV PATH="${NGS_TOOLS_DIR}:$PATH"

WORKDIR "${NGS_TOOLS_DIR}"

# bcl2fastq
RUN wget -q "https://burtonlab.s3.amazonaws.com/software/bcl2fastq2_0v2.20.0.422-2_amd64.deb" \ 
    && dpkg -i "bcl2fastq2_0v2.20.0.422-2_amd64.deb" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# bases2fastq
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/bases2fastq-1.4.0.tar.gz" \
    && tar xzvf "bases2fastq-1.4.0.tar.gz" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# FastQC
RUN wget -q "http://burtonlab.s3.amazonaws.com/software/fastqc_v0.11.9.zip" \
    && unzip "fastqc_v0.11.9.zip" \
    && chmod +x "${NGS_TOOLS_DIR}/FastQC/fastqc" \
    && ln -s "${NGS_TOOLS_DIR}/FastQC/fastqc" "/usr/local/bin/fastqc" \
    && fix-permissions "/home/${NB_USER}"

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
RUN wget -q "http://burtonlab.s3.amazonaws.com/software/zlib1211.zip" \
    && unzip zlib1211.zip \ 
    && cd "${NGS_TOOLS_DIR}/zlib-1.2.11" \
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
RUN wget -q "https://burtonlab.s3.amazonaws.com/software/cellranger-7.1.0.tar.gz" \
    && tar xzvf "cellranger-7.1.0.tar.gz" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}" 
RUN rm -rf "cellranger-7.1.0.tar.gz"

ENV PATH="${NGS_TOOLS_DIR}/cellranger-7.1.0/bin:$PATH"

# Parse pipeline
RUN wget -q "https://brineylab.s3.amazonaws.com/tools/parse-pipeline-1.2.0.tar.gz" \
    && tar xzvf "parse-pipeline-1.2.0.tar.gz" \
    && cd "${NGS_TOOLS_DIR}/parse-pipeline-1.2.0" \
    && bash ./install_dependencies_conda.sh -i -y \
    && bash ./install_TCR.sh -i -y \
    && python3 -m pip install . --no-cache-dir \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

ENV PATH="${NGS_TOOLS_DIR}/parse-pipeline-1.2.0:$PATH"

# IgDiscover
RUN cd "${NGS_TOOLS_DIR}" \
    && git clone "https://gitlab.com/gkhlab/igdiscover22.git" \
    && cd igdiscover22 \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"

USER ${NB_UID}

RUN cd "${NGS_TOOLS_DIR}/igdiscover22" \
    && conda env create -f environment.yml -n igdiscover \
    && source activate igdiscover \
    && python3 -m pip install --no-deps -e . \
    && python3 -m pip install networkx==2.1 \
    && conda deactivate


# ANARCI
USER ${NB_UID}

RUN cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/oxpig/ANARCI" \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"
    
RUN mamba install -c conda-forge biopython -y \
    && mamba install -c bioconda hmmer=3.3.2 -y \
    && cd "${NGS_TOOLS_DIR}/ANARCI" \
    && python3 setup.py install \
    && fix-permissions "${NGS_TOOLS_DIR}" \
    && fix-permissions "/home/${NB_USER}"


USER ${NB_UID}

WORKDIR "${HOME}"





