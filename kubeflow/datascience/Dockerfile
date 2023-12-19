ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/kubeflow-jupyter
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# set shell to bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# R dependencies
RUN apt update --yes \
   && apt install --yes --no-install-recommends \
   fonts-dejavu \
   gcc \
   gfortran \
   && apt clean && rm -rf /var/lib/apt/lists/*

RUN apt update --yes \
   && apt install --yes --no-install-recommends \
   "openjdk-${OPENJDK_VERSION}-jre-headless" \
   ca-certificates-java \
   && apt clean && rm -rf /var/lib/apt/lists/*

# Configure IPython system-wide
COPY ipython_kernel_config.py "/etc/ipython/"
RUN chown -R ${NB_USER}:users /etc/ipython/

# switch to NB_USER to install python packages
USER ${NB_UID}

# ab[x] and dependencies
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/abx_pip.txt \
   && fix-permissions ${REQUIREMENTS_DIR}
RUN python3 -m pip install -r ${REQUIREMENTS_DIR}/abx_pip.txt

# R packages including IRKernel which gets installed globally.
# r-e1071: dependency of the caret R package
RUN wget --directory-prefix=${REQUIREMENTS_DIR} https://raw.githubusercontent.com/briney/containers/main/requirements/r_mamba.txt \
   && fix-permissions ${REQUIREMENTS_DIR}
RUN mamba install --yes --file ${REQUIREMENTS_DIR}/r_mamba.txt \
   && mamba clean --all -f -y \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}"

# Julia installation
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
# Check https://julialang.org/downloads/
ARG julia_version="1.8.5"

USER root

# # install Julia packages in /opt/julia instead of ${HOME}
# ENV JULIA_DEPOT_PATH=/opt/julia \
#    JULIA_PKGDIR=/opt/julia \
#    JULIA_VERSION="${julia_version}"

# WORKDIR /tmp

# RUN set -x \
#    && julia_arch=$(uname -m) \
#    && julia_short_arch="${julia_arch}" \
#    && if [ "${julia_short_arch}" == "x86_64" ]; then \
#    julia_short_arch="x64"; \
#    fi; \
#    julia_installer="julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" \
#    && julia_major_minor=$(echo "${JULIA_VERSION}" | cut -d. -f 1,2) \
#    && mkdir "/opt/julia-${JULIA_VERSION}" \
#    && wget -q "https://julialang-s3.julialang.org/bin/linux/${julia_short_arch}/${julia_major_minor}/${julia_installer}" \
#    && tar xzf "${julia_installer}" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1 \
#    && rm "${julia_installer}" \
#    && ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

# # Show Julia where conda libraries are \
# RUN mkdir /etc/julia \
#    && echo "push!(Libdl.DL_LOAD_PATH, \"${CONDA_DIR}/lib\")" >> /etc/julia/juliarc.jl \
#    # Create JULIA_PKGDIR \
#    && mkdir "${JULIA_PKGDIR}" \
#    && chown "${NB_USER}" "${JULIA_PKGDIR}" \
#    && fix-permissions "${JULIA_PKGDIR}"

# # Add Julia packages.
# # Install IJulia as jovyan and then move the kernelspec out
# # to the system share location. Avoids problems with runtime UID change not
# # taking effect properly on the .local folder in the jovyan home dir.
# USER ${NB_UID}

# RUN julia -e 'import Pkg; Pkg.update()' \
#    && julia -e 'import Pkg; Pkg.add("HDF5")' \
#    && julia -e 'using Pkg; pkg"add IJulia"; pkg"precompile"' \
#    # move kernelspec out of home \
#    && mv "${HOME}/.local/share/jupyter/kernels/julia"* "${CONDA_DIR}/share/jupyter/kernels/" \
#    && chmod -R go+rx "${CONDA_DIR}/share/jupyter" \
#    && rm -rf "${HOME}/.local"

WORKDIR "${HOME}"

