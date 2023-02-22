ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/kubeflow-python
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# set shell to bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# install -- node.js
RUN curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
   # && echo "deb https://deb.nodesource.com/node_14.x focal main" > /etc/apt/sources.list.d/nodesource.list \
   && echo "deb https://deb.nodesource.com/node_16.x jammy main" > /etc/apt/sources.list.d/nodesource.list \
   && apt --yes update \
   && apt --yes install --no-install-recommends \
   nodejs \
   # for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
   # build-essential \
   # for latex labels
   cm-super \
   dvipng \
   # for matplotlib anim
   ffmpeg \
   && apt clean \
   && rm -rf /var/lib/apt/lists/*


# # setup environment for conda
# # ENV CONDA_DIR=/opt/conda
# # ENV PATH="${CONDA_DIR}/bin:${PATH}"
# RUN mkdir -p ${CONDA_DIR} \
#    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc \
#    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile \
#    && echo "conda activate base" >> ${HOME}/.bashrc \
#    && echo "conda activate base" >> /etc/profile \
#    && chown -R ${NB_USER}:users ${CONDA_DIR} \
#    && chown -R ${NB_USER}:users ${HOME}

# switch to NB_UID for installs
USER ${NB_UID}

# # Pin python version here, or set it to "default"
# ARG PYTHON_VERSION=3.9
# ARG PIP_VERSION="default"

# # Download and install Micromamba, and initialize Conda prefix.
# #    <https://github.com/mamba-org/mamba#micromamba>
# #    Similar projects using Micromamba:
# #     - Micromamba-Docker: <https://github.com/mamba-org/micromamba-docker>
# #     - repo2docker: <https://github.com/jupyterhub/repo2docker>
# # Install Python, Mamba, pip and jupyter_core
# # Cleanup temporary files and remove Micromamba
# # Correct permissions
# # Do all this in a single RUN command to avoid duplicating all of the
# # files across image layers when the permissions change
# COPY --chown="${NB_UID}:${NB_GID}" initial-condarc "${CONDA_DIR}/.condarc"
# WORKDIR /tmp
# RUN set -x && \
#    arch=$(uname -m) && \
#    if [ "${arch}" = "x86_64" ]; then \
#    # Should be simpler, see <https://github.com/mamba-org/mamba/issues/1437>
#    arch="64"; \
#    fi && \
#    wget -qO /tmp/micromamba.tar.bz2 \
#    "https://micromamba.snakepit.net/api/micromamba/linux-${arch}/latest" && \
#    tar -xvjf /tmp/micromamba.tar.bz2 --strip-components=1 bin/micromamba && \
#    rm /tmp/micromamba.tar.bz2 && \
#    PYTHON_SPECIFIER="python=${PYTHON_VERSION}" && \
#    if [[ "${PYTHON_VERSION}" == "default" ]]; then PYTHON_SPECIFIER="python"; fi && \
#    echo "installing ${PYTHON_SPECIFIER}" && \
#    PIP_SPECIFIER="pip=${PIP_VERSION}" && \
#    if [[ "${PIP_VERSION}" == "default" ]]; then PIP_SPECIFIER="pip"; fi && \
#    echo "installing ${PIP_SPECIFIER}" && \
#    # Install the packages
#    ./micromamba install \
#    --root-prefix="${CONDA_DIR}" \
#    --prefix="${CONDA_DIR}" \
#    --yes \
#    "${PYTHON_SPECIFIER}" \
#    "${PIP_SPECIFIER}" \
#    'mamba' \
#    'jupyter_core' && \
#    rm micromamba && \
#    # Pin major.minor version of python
#    mamba list python | grep '^python ' | tr -s ' ' | cut -d ' ' -f 1,2 >> "${CONDA_DIR}/conda-meta/pinned" && \
#    mamba clean --all -f -y

# jupyter
RUN mamba install --yes \
   'ipykernel' \
   'jupyterlab' \
   'notebook' \
   && jupyter lab --generate-config \
   && mamba clean --all -f -y \
   && npm cache clean --force \
   && jupyter lab clean \
   && rm -rf ${HOME}/.cache/yarn

# RUN mamba install --quiet --yes \
#    # kubeflow
#    'kfp' \
#    'kfp-server-api' \
#    # scipy
#    'altair' \
#    'beautifulsoup4' \
#    'black' \
#    'bokeh' \
#    'bottleneck' \
#    'cloudpickle' \
#    'conda-forge::blas=*=openblas' \
#    'cython' \
#    'dask' \
#    'dill' \
#    'h5py' \
#    'ipympl'\
#    'ipywidgets' \
#    'matplotlib-base' \
#    'natsort' \
#    'numba' \
#    'numexpr' \ 
#    'openpyxl' \
#    'pandas' \
#    'patsy' \
#    'protobuf' \
#    'pytables' \
#    'scikit-bio' \
#    'scikit-image' \
#    'scikit-learn' \
#    'scipy' \
#    'seaborn' \
#    'sqlalchemy' \
#    'statsmodels' \
#    'sympy' \
#    'widgetsnbextension'\
#    'xlrd'

# RUN python3 -m pip install jsonschema[format-nongpl]

RUN python3 -m pip install \
   'jupyterlab-code-formatter' \
   'jupyterlab-git' \
   # 'jupyterlab-fasta' \
   # 'jupyterlab-system-monitor' \
   # 'jupyterlab_tensorboard' \
   'nbdime'
# 'elyra[all]'

# # Install kfserving, which needs pip not mamba
# RUN python3 -m pip install kfserving

# # install Kale
# RUN python3 -m pip install markupsafe==2.0.1 && \
#    python3 -m pip install kubeflow-kale && \
#    jupyter labextension install kubeflow-kale-labextension


# install elyra
# RUN mamba install --yes jsonschema-with-format-nongpl
# RUN python3 -m pip install \
#    "elyra-pipeline-editor-extension" 
# RUN mamba install --yes elyra[all]



# Install extensions which do not have pip or conda packages at the moment
WORKDIR /tmp
RUN git clone https://github.com/PAIR-code/facets.git && \
   jupyter nbextension install facets/facets-dist/ --sys-prefix && \
   rm -rf /tmp/facets
# jupyter labextension install jupyterlab_voyager && \
# jupyter labextension install nbgather

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot"

# need to be root again to do some s6 stuff
USER root

# s6 - copy scripts
COPY s6/ /etc
RUN chown -R ${NB_USER}:users /etc/s6

# # s6 - 01-copy-tmp-home
# RUN mkdir -p /tmp_home \
#    && cp -r ${HOME} /tmp_home \
#    && chown -R ${NB_USER}:users /tmp_home

USER ${NB_UID}

WORKDIR ${HOME}

EXPOSE 8888

ENTRYPOINT ["/init"]
