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

# Runtime config
ENV PIP_NO_CACHE_DIR=1 \
    JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT


# ----------------------------
#      Jupyter Packages
# ----------------------------
# installs JupyterLab, Jupyter Notebook, and JupyterHub
# generates a Jupyter Server config
WORKDIR /tmp

# nodejs is required for JupyterLab extension building
RUN mamba install --yes 'nodejs>=20.0.0' \
    && mamba clean --all -f -y

COPY ./requirements/jupyter_pip.txt /tmp/jupyter_pip.txt
RUN uv pip install --no-cache --system -r /tmp/jupyter_pip.txt \
    && jupyter server --generate-config \
    && jupyter lab clean \
    && npm cache clean --force


# ----------------------------
#    JupyterLab Extensions
# ----------------------------
# JupyterLab NVdashboard (nightly for jupyterlab >= 4.0)
# https://github.com/rapidsai/jupyterlab-nvdashboard
RUN mamba install --yes -c rapidsai-nightly -c conda-forge jupyterlab-nvdashboard \
    && mamba clean --all -f -y


# ----------------------------
#    Container Startup
# ----------------------------
COPY ./jupyterhub/runtime/start-notebook.py ./jupyterhub/runtime/start-singleuser.py /usr/local/bin/
COPY ./jupyterhub/runtime/jupyter_server_config.py ./jupyterhub/runtime/docker_healthcheck.py /etc/jupyter/

HEALTHCHECK --interval=3s --timeout=1s --start-period=3s --retries=3 \
    CMD /etc/jupyter/docker_healthcheck.py || exit 1

# Add R mimetype option to specify how the plot returns from R to the browser
COPY ./jupyterhub/runtime/Rprofile.site /opt/conda/lib/R/etc/


# ----------------------------
#         Final Setup
# ----------------------------
WORKDIR /home/jovyan

ENTRYPOINT ["tini", "--"]
CMD ["start-notebook.py"]
