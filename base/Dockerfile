# Use the respective Makefile to build the image.
FROM ubuntu:22.04

# Images were initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and were subsequently modified by me.
ARG OWNER=brineylab
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# args
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# software versions
ARG KUBECTL_ARCH="amd64"
ARG KUBECTL_VERSION=v1.21.0
ARG S6_ARCH="amd64"
# renovate: datasource=github-tags depName=just-containers/s6-overlay versioning=loose
ARG S6_VERSION=v2.2.0.3

# set shell to bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# install - useful linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get --yes update \
    && apt-get --yes install --no-install-recommends \
    apt-transport-https \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    fonts-liberation \
    git \
    gnupg \
    gnupg2 \
    htop \
    less \
    locales \
    lsb-release \
    nano \
    openssh-client \
    pandoc \
    run-one \
    software-properties-common \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    tmux \
    tzdata \
    unzip \
    vim \
    wget \
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install - s6 overlay
RUN export GNUPGHOME=/tmp/ \
    && curl -sL "https://github.com/just-containers/s6-overlay/releases/download/${S6_VERSION}/s6-overlay-${S6_ARCH}-installer" -o /tmp/s6-overlay-${S6_VERSION}-installer \
    && curl -sL "https://github.com/just-containers/s6-overlay/releases/download/${S6_VERSION}/s6-overlay-${S6_ARCH}-installer.sig" -o /tmp/s6-overlay-${S6_VERSION}-installer.sig \
    && gpg --keyserver keys.gnupg.net --keyserver pgp.surfnet.nl --recv-keys 6101B2783B2FD161 \
    && gpg -q --verify /tmp/s6-overlay-${S6_VERSION}-installer.sig /tmp/s6-overlay-${S6_VERSION}-installer \
    && chmod +x /tmp/s6-overlay-${S6_VERSION}-installer \
    && /tmp/s6-overlay-${S6_VERSION}-installer / \
    && rm /tmp/s6-overlay-${S6_VERSION}-installer.sig /tmp/s6-overlay-${S6_VERSION}-installer

# install - kubectl
RUN curl -sL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${KUBECTL_ARCH}/kubectl" -o /usr/local/bin/kubectl \
    && curl -sL "https://dl.k8s.io/${KUBECTL_VERSION}/bin/linux/${KUBECTL_ARCH}/kubectl.sha256" -o /tmp/kubectl.sha256 \
    && echo "$(cat /tmp/kubectl.sha256) /usr/local/bin/kubectl" | sha256sum --check \
    && rm /tmp/kubectl.sha256 \
    && chmod +x /usr/local/bin/kubectl

# configure environment
ENV NB_USER="${NB_USER}" \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID}
# paths
ENV PATH="${CONDA_DIR}/bin:${PATH}" \
    HOME="/home/${NB_USER}" \
    CONDA_DIR=/opt/conda \
    NB_PREFIX="/"

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
    # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
    echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc

# create user and set required ownership
RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} \
    && mkdir -p ${HOME} \
    && chown -R ${NB_USER}:users ${HOME} \
    && chown -R ${NB_USER}:users /usr/local/bin \
    && chown -R ${NB_USER}:users /etc/s6

# set locale configs
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

USER $NB_UID
