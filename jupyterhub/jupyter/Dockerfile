ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/jupyterhub-base
FROM $BASE_IMG

# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified and maintained by the Briney Lab.
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed 
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license


# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \ 
        # for latex labels
        cm-super \ 
        dvipng \ 
        # for matplotlib anim
        ffmpeg \ 
        fonts-liberation \ 
        git \ 
        # less is needed to run help in R
        # see: https://github.com/jupyter/docker-stacks/issues/1588
        less \ 
        openssh-client \ 
        # - pandoc is used to convert notebooks to html files
        #   it's not present in aarch64 ubuntu image, so we install it here
        pandoc \
        # - run-one - a wrapper script that runs no more
        #   than one unique  instance  of  some  command with a unique set of arguments,
        #   we use `run-one-constantly` to support `RESTARTABLE` option
        run-one \ 
        # nbconvert dependencies
        # https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex
        texlive-xetex \ 
        texlive-fonts-recommended \ 
        texlive-plain-generic \ 
        tzdata \ 
        # Enable clipboard on Linux host systems
        xclip \ 
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install Jupyter Notebook, Lab, and Hub
# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
WORKDIR /tmp
RUN mamba install --quiet --yes \
        'notebook' \
        'jupyterhub' \
        'jupyterlab' \
    && jupyter notebook --generate-config \
    && mamba clean --all -f -y \
    && npm cache clean --force \
    && jupyter lab clean \
    && rm -rf "/home/${NB_USER}/.cache/yarn" \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

# Configure container startup
CMD ["start-notebook.sh"]

# Copy local files as late as possible to avoid cache busting
COPY start-notebook.sh start-singleuser.sh /usr/local/bin/ 

# Currently need to have both jupyter_notebook_config and jupyter_server_config to support classic and lab
COPY jupyter_server_config.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root

# Legacy for Jupyter Notebook Server, see: [#1205](https://github.com/jupyter/docker-stacks/issues/1205)
RUN sed -re "s/c.ServerApp/c.NotebookApp/g" \
    /etc/jupyter/jupyter_server_config.py > /etc/jupyter/jupyter_notebook_config.py \
    && fix-permissions /etc/jupyter/

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server` and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK  --interval=5s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -O- --no-verbose --tries=1 --no-check-certificate \
    http${GEN_CERT:+s}://localhost:${JUPYTER_PORT}${JUPYTERHUB_SERVICE_PREFIX:-/}api || exit 1


USER ${NB_UID}

WORKDIR "${HOME}"

# Add R mimetype option to specify how the plot returns from R to the browser
COPY --chown=${NB_UID}:${NB_GID} Rprofile.site /opt/conda/lib/R/etc/

# Install Python 3 packages
RUN mamba install --quiet --yes \
        'altair' \
        'beautifulsoup4' \
        'black' \
        'bokeh' \
        'bottleneck' \
        'cloudpickle' \
        'conda-forge::blas=*=openblas' \
        'cython' \
        'dask' \
        'dill' \
        'h5py' \
        'ipympl'\
        'ipywidgets' \
        # Temporary fix for: https://github.com/jupyter/docker-stacks/issues/1851
        'jupyter_server>=2.0.0' \
        'matplotlib-base' \
        'natsort' \
        'numba' \
        'numexpr' \
        'openpyxl' \
        'pandas' \
        'patsy' \
        'protobuf' \
        'pytables' \
        'scikit-bio' \
        'scikit-image' \
        'scikit-learn' \
        'scipy' \
        'seaborn' \
        'sqlalchemy' \
        'statsmodels' \
        'sympy' \
        'widgetsnbextension'\
        'xlrd' \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN python3 -m pip install \
   'jupyterlab-code-formatter' \
   'jupyterlab-git' \
   # 'jupyterlab-fasta' \
   # 'jupyterlab-system-monitor' \
   # 'jupyterlab_tensorboard' \
   'nbdime'

# Install facets which does not have a pip or conda package at the moment
WORKDIR /tmp
RUN git clone https://github.com/PAIR-code/facets.git \
    && jupyter nbextension install facets/facets-dist/ --sys-prefix \
    && rm -rf /tmp/facets \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" \
    && fix-permissions "/home/${NB_USER}"

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

WORKDIR "${HOME}"
