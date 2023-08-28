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

#-----------------
#     SPARK
#-----------------

# Spark dependencies
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
ARG spark_version="3.3.2"
ARG hadoop_version="3"
ARG scala_version
ARG spark_checksum="4cd2396069fbe0f8efde2af4fd301bf46f8c6317e9dea1dd42a405de6a38380635d49b17972cb92c619431acece2c3af4c23bfdf193cedb3ea913ed69ded23a1"
ARG openjdk_version="17"

ENV APACHE_SPARK_VERSION="${spark_version}" \
    HADOOP_VERSION="${hadoop_version}"

RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    "openjdk-${openjdk_version}-jre-headless" \
    ca-certificates-java \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# Spark installation
WORKDIR /tmp

# You need to use https://archive.apache.org/dist/ website if you want to download old Spark versions
# But it seems to be slower, that's why we use recommended site for download
RUN if [ -z "${scala_version}" ]; then \
    wget -qO "spark.tgz" "https://dlcdn.apache.org/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"; \
    else \
    wget -qO "spark.tgz" "https://dlcdn.apache.org/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}-scala${scala_version}.tgz"; \
    fi && \
    echo "${spark_checksum} *spark.tgz" | sha512sum -c - \
    && tar xzf "spark.tgz" -C /usr/local --owner root --group root --no-same-owner \
    && rm "spark.tgz"

# Configure Spark
ENV SPARK_HOME=/usr/local/spark
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH="${PATH}:${SPARK_HOME}/bin"

RUN if [ -z "${scala_version}" ]; then \
    ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}"; \
    else \
    ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}-scala${scala_version}" "${SPARK_HOME}"; \
    fi \
    # Add a link in the before_notebook hook in order to source automatically PYTHONPATH && \
    && mkdir -p /usr/local/bin/before-notebook.d \
    && ln -s "${SPARK_HOME}/sbin/spark-config.sh" /usr/local/bin/before-notebook.d/spark-config.sh

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

ARG CODESERVER_VERSION=v4.9.1

# install - code-server
RUN curl -sL "https://github.com/cdr/code-server/releases/download/${CODESERVER_VERSION}/code-server_${CODESERVER_VERSION/v/}_amd64.deb" -o /tmp/code-server.deb \
    && dpkg -i /tmp/code-server.deb \
    && rm -f /tmp/code-server.deb

# switch to NB_USER to install python packages
USER ${NB_UID}

RUN mamba install --quiet --yes \
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

USER root

# install ubuntu desktop GUI
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    # "ubuntu-desktop" \
    "dbus-x11" \ 
    "libgl1-mesa-glx" \
    "xorg" \ 
    "xfce4" \
    "xfce4-panel" \ 
    "xfce4-session" \ 
    "xfce4-settings" \ 
    "xubuntu-icon-theme" \ 
    # these are required to add-apt-repository
    # "software-properties-common" \
    # "gpg" \
    # "gpg-agent" \
    && apt-get remove --yes --quiet light-locker \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# # upgrade to xfce 4.18
# RUN add-apt-repository ppa:xubuntu-dev/staging \
#     && apt-get update \
#     && apt-get upgrade --yes \
#     && apt-get clean \ 
#     && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# install plugin for accessing the desktop
# from inside JupyterHub
RUN python3 -m pip install \
    "jupyter-remote-desktop-proxy"

# install websockify
RUN mamba install --quiet --yes \
    "websockify" \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


#-----------------------------
#     ab[x] and R packages
#-----------------------------

# # baltic still has a bug that crashes the pip install from PyPI
# WORKDIR /tmp
# RUN git clone https://github.com/evogytis/baltic \
#     && cd baltic \
#     && python3 -m pip install ./
# WORKDIR "${HOME}"

# ab[x] and dependencies
# COPY ../../requirements/abx_pip.txt /tmp
# RUN fix-permissions /tmp
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/abx_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}

RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/abx_pip.txt

# find spark and initialize
RUN python3 -c "import findspark; findspark.init()"


# R packages including IRKernel which gets installed globally.
# r-e1071: dependency of the caret R package
# COPY ../../requirements/r_mamba.txt /tmp
# RUN fix-permissions /tmp
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/r_mamba.txt \
    && fix-permissions ${REQUIREMENTS_DIR}

RUN mamba install --yes --file ${REQUIREMENTS_DIR}/r_mamba.txt \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

#-----------------
#     JULIA
#-----------------

USER root

# Julia installation
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
# Check https://julialang.org/downloads/
ARG julia_version="1.8.5"

# Julia dependencies
# install Julia packages in /opt/julia instead of ${HOME}
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia \
    JULIA_VERSION="${julia_version}"

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


USER root

ENV NGS_TOOLS_DIR="/tools"

RUN mkdir "${NGS_TOOLS_DIR}" \
    && chown "${NB_USER}" "${NGS_TOOLS_DIR}" \
    && fix-permissions "${NGS_TOOLS_DIR}"

WORKDIR "${NGS_TOOLS_DIR}"

# bcl2fastq
RUN wget -q "https://burtonlab.s3.amazonaws.com/software/bcl2fastq2_0v2.20.0.422-2_amd64.deb" \ 
    && dpkg -i "bcl2fastq2_0v2.20.0.422-2_amd64.deb" \
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

ENV PATH="${NGS_TOOLS_DIR}/cellranger-7.1.0/bin:$PATH"


USER ${NB_UID}

WORKDIR "${HOME}"





