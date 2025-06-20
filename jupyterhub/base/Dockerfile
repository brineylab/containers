# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks/tree/main/images/docker-stacks-foundation
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

# NB user
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Python version
ARG PYTHON_VERSION=3.11

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
        # - `netbase` provides /etc/{protocols,rpc,services}, part of POSIX
        #   and required by various C functions like getservbyname and getprotobyname
        #   https://github.com/jupyter/docker-stacks/pull/2129
        netbase \
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
    # More information in: https://github.com/jupyter/docker-stacks/pull/2047
    # and docs: https://docs.conda.io/projects/conda/en/latest/dev-guide/deep-dives/activation.html
    echo 'eval "$(conda shell.bash hook)"' >> /etc/skel/.bashrc

# Delete existing user with UID="${NB_UID}" if it exists
# hadolint ignore=SC2046
RUN if grep -q "${NB_UID}" /etc/passwd; then \
        userdel --remove $(id -un "${NB_UID}"); \
    fi

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

# macOS Rosetta virtualization creates junk directory which gets owned by root further up.
# It'll get re-created, but as USER runner after the next directive so hopefully should not cause permission issues.
# More info: https://github.com/jupyter/docker-stacks/issues/2296
RUN rm -rf "/home/${NB_USER}/.cache/"


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
    && if [ "${arch}" = "x86_64" ]; then \
        # Should be simpler, see <https://github.com/mamba-org/mamba/issues/1437>
        arch="64"; \
    fi \
    # https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#linux-and-macos
    && wget --progress=dot:giga -O - \
        "https://micro.mamba.pm/api/micromamba/linux-${arch}/latest" | tar -xvj bin/micromamba \
    && PYTHON_SPECIFIER="python=${PYTHON_VERSION}" \
    && if [[ "${PYTHON_VERSION}" == "default" ]]; then PYTHON_SPECIFIER="python"; fi \
    # Install the packages
    && ./bin/micromamba install \
        --root-prefix="${CONDA_DIR}" \
        --prefix="${CONDA_DIR}" \
        --yes \
        'jupyter_core' \
        'conda' \
        'mamba' \
        "${PYTHON_SPECIFIER}" \
    && rm -rf /tmp/bin/ \
    # Pin major.minor version of python
    # https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning
    && mamba list --full-name 'python' | awk 'END{sub("[^.]*$", "*", $2); print $1 " " $2}' >> "${CONDA_DIR}/conda-meta/pinned" \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


# ----------------------------
#         Directories
# ----------------------------
# Make requirements directory
# And setup work directory for backward-compatibility
RUN mkdir -p ${REQUIREMENTS_DIR} \
    && mkdir "/home/${NB_USER}/work" \
    && mkdir "/home/${NB_USER}/codeserver" \
    && fix-permissions ${REQUIREMENTS_DIR} \
    && fix-permissions "/home/${NB_USER}"

# Create dirs for startup hooks
USER root
RUN mkdir /usr/local/bin/start-notebook.d \
    && mkdir /usr/local/bin/before-notebook.d
COPY ./jupyterhub/base/10activate-conda-env.sh /usr/local/bin/before-notebook.d/

# macOS Rosetta virtualization creates junk directory which gets owned by root further up.
# It'll get re-created, but as USER runner after the next directive so hopefully should not cause permission issues.
# More info: https://github.com/jupyter/docker-stacks/issues/2296
RUN rm -rf "/home/${NB_USER}/.cache/"


# ----------------------------
#         Final Setup
# ----------------------------
# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR "${HOME}"

# Configure container startup
COPY ./jupyterhub/base/start.sh ./jupyterhub/base/run-hooks.sh /usr/local/bin/
ENTRYPOINT ["tini", "-g", "--", "start.sh"]
