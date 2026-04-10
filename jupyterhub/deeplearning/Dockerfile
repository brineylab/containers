ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/deeplearning
FROM $BASE_IMG

# This image was initially created by the Jupyter Development Team:
# https://github.com/jupyter/docker-stacks
# and was subsequently modified and maintained by the Briney Lab.
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license


# ----------------------------
#      Jupyter Packages
# ----------------------------
# installs JupyterLab, Jupyter Notebook, JupyterHub and NBClassic
# generates a Jupyter Server config
USER ${NB_UID}
WORKDIR /tmp

# nodejs is required for JupyterLab extension building
RUN mamba install --yes 'nodejs>=20.0.0' \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}"

# Install as root so entry points (jupyterhub-singleuser) go to /usr/local/bin
USER root
COPY --chown=${NB_UID}:${NB_GID} ./requirements/jupyter_pip.txt ${REQUIREMENTS_DIR}/jupyter_pip.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r "${REQUIREMENTS_DIR}/jupyter_pip.txt" \
    && jupyter server --generate-config \
    && jupyter lab clean \
    && npm cache clean --force \
    && rm -rf "/home/${NB_USER}/.cache/yarn" \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


# ----------------------------
#    JupyterLab Extensions
# ----------------------------
ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

# mercury is installed as a Python package (for notebook widget use)
# but its server extension breaks JupyterHub routing
RUN sed -i 's/"mercury_app": true/"mercury_app": false/' /usr/local/etc/jupyter/jupyter_server_config.d/mercury_app.json

# JupyterLab NVdashboard (nightly for jupyterlab >= 4.0)
# https://github.com/rapidsai/jupyterlab-nvdashboard
RUN mamba install --yes -c rapidsai-nightly -c conda-forge jupyterlab-nvdashboard \
    && mamba clean --all -f -y


# ----------------------------
#    Container Startup
# ----------------------------
# Copy local files as late as possible to avoid cache busting
USER ${NB_UID}
COPY ./jupyterhub/runtime/start-notebook.py ./jupyterhub/runtime/start-singleuser.py /usr/local/bin/
COPY ./jupyterhub/runtime/jupyter_server_config.py ./jupyterhub/runtime/docker_healthcheck.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root
RUN fix-permissions /etc/jupyter/

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server` and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK --interval=3s --timeout=1s --start-period=3s --retries=3 \
    CMD /etc/jupyter/docker_healthcheck.py || exit 1

RUN rm -rf "/home/${NB_USER}/.cache/"

# Add R mimetype option to specify how the plot returns from R to the browser
USER ${NB_UID}
COPY --chown=${NB_UID}:${NB_GID} ./jupyterhub/runtime/Rprofile.site /opt/conda/lib/R/etc/


# ----------------------------
#         Final Setup
# ----------------------------
USER ${NB_UID}
WORKDIR "${HOME}"

# Configure container startup
ENTRYPOINT ["tini", "--"]
CMD ["start-notebook.py"]
