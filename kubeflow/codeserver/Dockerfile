ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/kubeflow-python
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# args - software versions
# renovate: datasource=github-tags depName=cdr/code-server versioning=semver
# ARG CODESERVER_VERSION=v4.3.0
ARG CODESERVER_VERSION=v4.9.1

# install - code-server
RUN curl -sL "https://github.com/cdr/code-server/releases/download/${CODESERVER_VERSION}/code-server_${CODESERVER_VERSION/v/}_amd64.deb" -o /tmp/code-server.deb \
    && dpkg -i /tmp/code-server.deb \
    && rm -f /tmp/code-server.deb


# switch to NB_USER to install python packages
USER ${NB_UID}

# ab[x] and dependencies
RUN python3 -m pip install \
    'anndata' \
    # 'baltic' \
    'biopython' \
    'celery' \
    'dnachisel' \
    'doubletdetection' \
    'ete3' \
    'fastcluster' \
    'harmonypy' \
    'humanize' \
    'leidenalg' \
    'logomaker' \
    'matplotlib' \
    'mnemonic' \
    'nwalign3' \
    'paramiko' \
    'prettytable' \
    'pyarrow' \
    'pycircos' \
    'pymongo' \
    'pytest' \
    'python-Levenshtein' \
    'pyyaml' \
    'sample-sheet' \
    'scanorama' \
    'scanpy' \
    'scrublet' \
    'scvelo' \
    'smart_open' \
    'tqdm' \
    'umap-learn' \
    # ab[x]
    'abutils' \
    'abstar' \
    'scab'


USER root

# s6 - copy scripts
# COPY s6/ /etc
# RUN chown -R ${NB_USER}:users /etc/s6
COPY --chown=${NB_USER}:users s6/ /etc

# # s6 - 01-copy-tmp-home
# RUN mkdir -p /tmp_home \
#     && cp -r ${HOME} /tmp_home \
#     && chown -R ${NB_USER}:users /tmp_home

USER $NB_UID

EXPOSE 8888

ENTRYPOINT ["/init"]
