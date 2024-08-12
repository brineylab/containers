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

# node.js setup script - version 20
# RUN curl -fsSL https://deb.nodesource.com/setup_20.x -o nodesource_setup.sh \
#    && bash -c 'DISTRO=jammy bash nodesource_setup.sh' \
#    && cat /etc/apt/sources.list.d/nodesource.list

# RUN apt --yes update \
#    && apt --yes install --no-install-recommends \
#    nodejs \
#    # for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
#    # build-essential \
#    # for latex labels
#    cm-super \
#    dvipng \
#    # for matplotlib anim
#    ffmpeg \
#    && apt clean \
#    && rm -rf /var/lib/apt/lists/*

RUN curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
    && echo "deb https://deb.nodesource.com/node_20.x jammy main" > /etc/apt/sources.list.d/nodesource.list \
    && cat /etc/apt/sources.list.d/nodesource.list

RUN apt --yes update \
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

RUN node -v

# jupyter
RUN wget --directory-prefix="${REQUIREMENTS_DIR}" https://raw.githubusercontent.com/briney/containers/main/requirements/jupyter_mamba.txt \
   && fix-permissions "${REQUIREMENTS_DIR}"

RUN mamba install --yes --file "${REQUIREMENTS_DIR}/jupyter_mamba.txt" \
   && jupyter notebook --generate-config \
   && mamba clean --all -f -y \
   && npm cache clean --force \
   && jupyter lab clean \
   && rm -rf "/home/${NB_USER}/.cache/yarn" \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}"

# switch to NB_UID for installs
USER ${NB_UID}

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

# Install jupyterlab extensions
RUN wget --directory-prefix="${REQUIREMENTS_DIR}" https://raw.githubusercontent.com/briney/containers/main/requirements/jupyter-extensions_pip.txt \
   && fix-permissions "${REQUIREMENTS_DIR}"
RUN python3 -m pip install -r "${REQUIREMENTS_DIR}/jupyter-extensions_pip.txt"

# RUN jupyter lab clean
RUN jupyter lab clean \
   && jupyter lab build --minimize=False

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot"

# need to be root again to do some s6 stuff
USER root

# s6 - copy scripts
COPY s6/ /etc
RUN chown -R ${NB_USER}:users /etc/s6
RUN chmod -R 755 /etc/s6

# # s6 - 01-copy-tmp-home
RUN mkdir -p /tmp_home \
   && cp -r ${HOME} /tmp_home \
   && chown -R ${NB_USER}:users /tmp_home

USER ${NB_UID}

WORKDIR ${HOME}

# EXPOSE 8888

ENTRYPOINT ["/init"]
