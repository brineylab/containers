ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/jupyterhub-base
FROM $BASE_IMG

# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks/blob/main/images/base-notebook/Dockerfile
# https://github.com/jupyter/docker-stacks/blob/main/images/minimal-notebook/Dockerfile
# https://github.com/jupyter/docker-stacks/blob/main/images/scipy-notebook/Dockerfile
# and was subsequently modified and maintained by the Briney Lab.
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed 
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license


# ----------------------------
#       OS dependencies
# ----------------------------
USER root
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
        # for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
        build-essential \
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


# ----------------------------
#       Mamba Packages
# ----------------------------
# installs JupyterLab, Jupyter Notebook, JupyterHub and NBClassic
# generates a Jupyter Server config
USER ${NB_UID}
WORKDIR /tmp
COPY --chown=${NB_UID}:${NB_GID} ./requirements/jupyter_mamba.txt ${REQUIREMENTS_DIR}/jupyter_mamba.txt
RUN mamba install --yes --file "${REQUIREMENTS_DIR}/jupyter_mamba.txt" \
    && jupyter server --generate-config \
    && mamba clean --all -f -y \
    && jupyter lab clean \
    && rm -rf "/home/${NB_USER}/.cache/yarn" \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


# ----------------------------
#    JupyterLab Extensions
# ----------------------------
ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

COPY --chown=${NB_UID}:${NB_GID} ./requirements/jupyter-extensions_pip.txt ${REQUIREMENTS_DIR}/jupyter-extensions_pip.txt
RUN python3 -m pip install -r "${REQUIREMENTS_DIR}/jupyter-extensions_pip.txt" \
    && jupyter lab clean \
    && fix-permissions "/home/${NB_USER}"


#----------------------
#     CODESERVER
#----------------------
# code-server args
ARG codeserver_version=v4.90.0
ENV CODESERVER_VERSION="${codeserver_version}" \
    CODE_WORKINGDIR="${HOME}/codeserver"

# install - code-server
USER root
RUN curl -sL "https://github.com/cdr/code-server/releases/download/${CODESERVER_VERSION}/code-server_${CODESERVER_VERSION/v/}_amd64.deb" -o /tmp/code-server.deb \
    && dpkg -i /tmp/code-server.deb \
    && rm -f /tmp/code-server.deb

# code-server extensions
USER ${NB_UID}
RUN code-server --install-extension ms-python.python \
                --install-extension ms-python.debugpy \
                --install-extension ms-python.black-formatter \
                --install-extension ms-toolsai.jupyter \
                --install-extension redhat.vscode-yaml \
                --install-extension redhat.vscode-xml


# ----------------------------
#    Container Startup
# ----------------------------
# Copy local files as late as possible to avoid cache busting
COPY ./jupyterhub/jupyter/start-notebook.py ./jupyterhub/jupyter/start-notebook.sh /usr/local/bin/
COPY ./jupyterhub/jupyter/start-singleuser.py ./jupyterhub/jupyter/start-singleuser.sh /usr/local/bin/
COPY ./jupyterhub/jupyter/jupyter_server_config.py ./jupyterhub/jupyter/docker_healthcheck.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root
RUN fix-permissions /etc/jupyter/

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server` and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK --interval=3s --timeout=1s --start-period=3s --retries=3 \
    CMD /etc/jupyter/docker_healthcheck.py || exit 1


# ----------------------------
#         Final Setup
# ----------------------------
# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR "${HOME}"

# Add R mimetype option to specify how the plot returns from R to the browser
COPY --chown=${NB_UID}:${NB_GID} ./jupyterhub/jupyter/Rprofile.site /opt/conda/lib/R/etc/

# Configure container startup
CMD ["start-notebook.py"]
