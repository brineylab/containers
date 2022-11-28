ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/base
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# args - software versions
# renovate: datasource=github-tags depName=cdr/code-server versioning=semver
ARG CODESERVER_VERSION=v4.3.0

# install - code-server
RUN curl -sL "https://github.com/cdr/code-server/releases/download/${CODESERVER_VERSION}/code-server_${CODESERVER_VERSION/v/}_amd64.deb" -o /tmp/code-server.deb \
    && dpkg -i /tmp/code-server.deb \
    && rm -f /tmp/code-server.deb

# s6 - copy scripts
COPY s6/ /etc
RUN chown -R ${NB_USER}:users /etc/s6

# s6 - 01-copy-tmp-home
RUN mkdir -p /tmp_home \
    && cp -r ${HOME} /tmp_home \
    && chown -R ${NB_USER}:users /tmp_home

USER $NB_UID

EXPOSE 8888

ENTRYPOINT ["/init"]
