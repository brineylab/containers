# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified by the Briney Lab.

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed 
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license

ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/base

FROM $BASE_IMG
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
# ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends \
    # - tini is installed as a helpful container entrypoint that reaps zombie
    #   processes and such of the actual executable we want to start, see
    #   https://github.com/krallin/tini#why-tini for details.
    tini \
    gnupg \
    gnupg2 \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Setup work directory for backward-compatibility
RUN mkdir "/home/${NB_USER}/work" \
    && mkdir "/home/${NB_USER}/codeserver" \
    && fix-permissions "/home/${NB_USER}"

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start.sh"]

# Copy local files as late as possible to avoid cache busting
COPY start.sh /usr/local/bin/

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

WORKDIR "${HOME}"