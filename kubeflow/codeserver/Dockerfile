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
# ARG CODESERVER_VERSION=v4.9.1

# install - code-server
RUN curl -sL "https://github.com/coder/code-server/releases/download/${CODESERVER_VERSION}/code-server_${CODESERVER_VERSION/v/}_amd64.deb" -o /tmp/code-server.deb \
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


# # switch to NB_USER to install python packages
# USER ${NB_UID}

# # baltic still has a bug that crashes the pip install from PyPI
# RUN git clone https://github.com/evogytis/baltic \
#     && cd baltic \
#     && python3 -m pip install ./

# ab[x] and dependencies
# COPY ../../requirements/abx_pip.txt /tmp
# RUN fix-permissions /tmp
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/abx_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}

RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/abx_pip.txt

# # ab[x] and dependencies
# RUN python3 -m pip install \
#     'anndata' \
#     # 'baltic' \
#     'biopython' \
#     'celery' \
#     'dnachisel' \
#     'doubletdetection' \
#     'ete3' \
#     'fastcluster' \
#     'harmonypy' \
#     'humanize' \
#     'leidenalg' \
#     'logomaker' \
#     'matplotlib' \
#     'mnemonic' \
#     'nwalign3' \
#     'paramiko' \
#     'prettytable' \
#     'pyarrow' \
#     'pycircos' \
#     'pymongo' \
#     'pytest' \
#     'python-Levenshtein' \
#     'pyyaml' \
#     'sample-sheet' \
#     'scanorama' \
#     'scanpy' \
#     'scrublet' \
#     'scvelo' \
#     'smart_open' \
#     'tqdm' \
#     'umap-learn' \
#     # ab[x]
#     'abutils' \
#     'abstar' \
#     'scab'


# pytorch
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/torch_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    -r ${REQUIREMENTS_DIR}/torch_pip.txt

# jax
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/jax_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    -r ${REQUIREMENTS_DIR}/jax_pip.txt

# other AI/ML packages (including 🤗)
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/ai-ml_pip.txt \
    && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/aiml_pip.txt


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
