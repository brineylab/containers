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
ENV CODESERVER_VERSION="${codeserver_version}"

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
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV HOME="/home/${NB_USER}"
ENV REQUIREMENTS_DIR="${HOME}/.requirements"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]


# ----------------------------
#       OS dependencies
# ----------------------------
USER root
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

# ----------------------------
#       User Setup
# ----------------------------
# Copy script to fix permissions
COPY --chown=${NB_UID}:${NB_GID} ./jupyterhub/base/fix-permissions /usr/local/bin/fix-permissions
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


# ----------------------------
#         Micromamba
# ----------------------------
# Download and install Micromamba, and initialize Conda prefix.
#   <https://github.com/mamba-org/mamba#micromamba>
#   Similar projects using Micromamba:
#     - Micromamba-Docker: <https://github.com/mamba-org/micromamba-docker>
#     - repo2docker: <https://github.com/jupyterhub/repo2docker>
USER ${NB_UID}
COPY --chown="${NB_UID}:${NB_GID}" ./jupyterhub/base/initial-condarc "${CONDA_DIR}/.condarc"
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


# ----------------------------
#         Directories
# ----------------------------
USER root

# Make requirements directory
# And work directory for backward-compatibility
RUN mkdir -p ${REQUIREMENTS_DIR} \
    && mkdir "/home/${NB_USER}/work" \
    && mkdir "/home/${NB_USER}/codeserver" \
    && fix-permissions ${REQUIREMENTS_DIR} \
    && fix-permissions "/home/${NB_USER}"


# ----------------------------
#         Final Setup
# ----------------------------
# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR "${HOME}"

# Configure container startup
COPY --chown=${NB_UID}:${NB_GID} ./jupyterhub/base/start.sh /usr/local/bin/
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start.sh"]
