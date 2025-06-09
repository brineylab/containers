# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified by the Briney Lab.

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed 
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license

# The default base image is Ubuntu 22.04 (jammy)
# https://hub.docker.com/_/ubuntu/tags?page=1&name=jammy

ARG BASE_IMG=ubuntu:22.04
FROM $BASE_IMG

ARG OWNER=brineylab
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# VERSIONS
ARG PYTHON_VERSION=3.11
ARG codeserver_version=v4.18.0
ARG spark_version="3.3.1"
ARG spark_checksum="769db39a560a95fd88b58ed3e9e7d1e92fb68ee406689fb4d30c033cb5911e05c1942dcc70e5ec4585df84e80aabbc272b9386a208debda89522efff1335c8ff"
ARG openjdk_version="17"
ARG hadoop_version="3"
ARG julia_version="1.8.5"
ENV CODESERVER_VERSION="${codeserver_version}" \
    APACHE_SPARK_VERSION="${spark_version}" \
    SPARK_CHECKSUM="${spark_checksum}" \
    OPENJDK_VERSION="${openjdk_version}" \
    HADOOP_VERSION="${hadoop_version}" \
    JULIA_VERSION="${julia_version}"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Root user
USER root

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --yes \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends \
        # - bzip2 is necessary to extract the micromamba executable.
        bzip2 \
        ca-certificates \
        locales \
        sudo \
        # - tini is installed as a helpful container entrypoint that reaps zombie
        #   processes and such of the actual executable we want to start, see
        #   https://github.com/krallin/tini#why-tini for details.
        tini \
        wget \
        # - additional apt packages beyond those in the original Jupyter dockerfile
        bash \
        bashtop \
        btop \ 
        build-essential \ 
        curl \ 
        htop \ 
        nano \
        nfs-common \ 
        openssh-server \ 
        pigz \ 
        screen \
        tmux \ 
        traceroute \ 
        unzip \ 
        vim \ 
        zip \ 
        gnupg \
        gnupg2 \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/* \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

# Configure environmental variables
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER="${NB_USER}" \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    PATH="${CONDA_DIR}/bin:${PATH}" \
    HOME="/home/${NB_USER}" \
    REQUIREMENTS_DIR="${HOME}/.requirements"

# Copy script to fix permissions
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
    # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
    echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc

# Create NB_USER with name jovyan user with UID=1000 and in the 'users' group
# Make sure these dirs are writable by the `users` group
# Grant sudo and add password
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

# Switch to notebook user for user-level setup
USER ${NB_UID}

# Download and install Micromamba, and initialize Conda prefix.
#   <https://github.com/mamba-org/mamba#micromamba>
#   Similar projects using Micromamba:
#     - Micromamba-Docker: <https://github.com/mamba-org/micromamba-docker>
#     - repo2docker: <https://github.com/jupyterhub/repo2docker>
COPY --chown="${NB_UID}:${NB_GID}" initial-condarc "${CONDA_DIR}/.condarc"
WORKDIR /tmp
RUN set -x \
    && arch=$(uname -m) \
    # Should be simpler, see <https://github.com/mamba-org/mamba/issues/1437>
    && if [ "${arch}" = "x86_64" ]; then arch="64"; fi \
    && wget -qO /tmp/micromamba.tar.bz2 --no-check-certificate \
        "https://micromamba.snakepit.net/api/micromamba/linux-${arch}/latest" \
    && tar -xvjf /tmp/micromamba.tar.bz2 --strip-components=1 bin/micromamba \
    && rm /tmp/micromamba.tar.bz2 \
    && PYTHON_SPECIFIER="python=${PYTHON_VERSION}" \
    && if [[ "${PYTHON_VERSION}" == "default" ]]; then PYTHON_SPECIFIER="python"; fi \
    # Install the packages (Python, Mamba and jupyter_core) & remove Micromamba
    && ./micromamba install \
        --root-prefix="${CONDA_DIR}" \
        --prefix="${CONDA_DIR}" \
        --yes \
        "${PYTHON_SPECIFIER}" \
        'mamba' \
        'jupyter_core' \
    && rm micromamba \
    # Pin major.minor version of python
    && echo "python ${PYTHON_VERSION}" >> "${CONDA_DIR}/conda-meta/pinned" \
    # Clean & correct permissions
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# Make requirements directory
ENV REQUIREMENTS_DIR="${HOME}/.requirements"
RUN mkdir -p ${REQUIREMENTS_DIR} \
    && fix-permissions ${REQUIREMENTS_DIR}

# Setup work directory for backward-compatibility
RUN mkdir "/home/${NB_USER}/work" \
    && mkdir "/home/${NB_USER}/codeserver" \
    && fix-permissions "/home/${NB_USER}"

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start.sh"]

# Copy local files as late as possible to avoid cache busting
COPY start.sh /usr/local/bin/
WORKDIR "${HOME}"