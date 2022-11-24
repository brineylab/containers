ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/jupyter
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
 && apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    gcc \
    gfortran \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# copy requirements.txt
COPY requirements.txt /tmp
RUN chown -R ${NB_USER}:users /tmp/requirements.txt

# switch to NB_USER to install python packages
USER ${NB_UID}

# scikit-bio is particularly fragile and 
# needs to be istalled separately with conda
RUN conda install -c conda-forge scikit-bio 

# install - requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
 && rm -f /tmp/requirements.txt

# R packages including IRKernel which gets installed globally.
# r-e1071: dependency of the caret R package
RUN conda install --quiet --yes \
    'r-base' \
    'r-ggplot2' \
    'r-irkernel' \
    'r-rcurl' \
    'r-sparklyr' \
    # 'r-caret' \
    # 'r-crayon' \
    # 'r-devtools' \
    # 'r-e1071' \
    # 'r-forecast' \
    # 'r-ggplot2' \
    # 'r-hexbin' \
    # 'r-htmltools' \
    # 'r-htmlwidgets' \
    # 'r-irkernel' \
    # 'r-nycflights13' \
    # 'r-randomforest' \
    # 'r-rcurl' \
    # 'r-rmarkdown' \
    # 'r-rodbc' \
    # 'r-rsqlite' \
    # 'r-shiny' \
    # 'r-sparklyr' \
    # 'r-tidymodels' \
    'r-tidyverse' \
    # 'rpy2' \
    # 'unixodbc' \
 && conda clean -a -f -y






# # Julia installation
# # Default values can be overridden at build time
# # (ARGS are in lower case to distinguish them from ENV)
# # Check https://julialang.org/downloads/
# ARG julia_version="1.8.3"

# USER root

# # install Julia packages in /opt/julia instead of ${HOME}
# ENV JULIA_DEPOT_PATH=/opt/julia \
#     JULIA_PKGDIR=/opt/julia \
#     JULIA_VERSION="${julia_version}"

# WORKDIR /tmp

# RUN set -x \
#  && julia_arch=$(uname -m) \
#  && julia_short_arch="${julia_arch}" \
#  && if [ "${julia_short_arch}" == "x86_64" ]; then \
#       julia_short_arch="x64"; \
#     fi; \
#     julia_installer="julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" \
#  && julia_major_minor=$(echo "${JULIA_VERSION}" | cut -d. -f 1,2) \
#  && mkdir "/opt/julia-${JULIA_VERSION}" \
#  && wget -q "https://julialang-s3.julialang.org/bin/linux/${julia_short_arch}/${julia_major_minor}/${julia_installer}" \
#  && tar xzf "${julia_installer}" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1 \
#  && rm "${julia_installer}" \
#  && ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

# # Show Julia where conda libraries are \
# RUN mkdir /etc/julia \
#  && echo "push!(Libdl.DL_LOAD_PATH, \"${CONDA_DIR}/lib\")" >> /etc/julia/juliarc.jl \
#  # Create JULIA_PKGDIR \
#  && mkdir "${JULIA_PKGDIR}" \
#  && chown "${NB_USER}" "${JULIA_PKGDIR}" \
#  && fix-permissions "${JULIA_PKGDIR}"



# # Add Julia packages.
# # Install IJulia as jovyan and then move the kernelspec out
# # to the system share location. Avoids problems with runtime UID change not
# # taking effect properly on the .local folder in the jovyan home dir.

# USER ${NB_UID}

# RUN julia -e 'import Pkg; Pkg.update()' \
#  && julia -e 'import Pkg; Pkg.add("HDF5")' \
#  && julia -e 'using Pkg; pkg"add IJulia"; pkg"precompile"' \
#  # move kernelspec out of home \
#  && mv "${HOME}/.local/share/jupyter/kernels/julia"* "${CONDA_DIR}/share/jupyter/kernels/" \
#  && chmod -R go+rx "${CONDA_DIR}/share/jupyter" \
#  && rm -rf "${HOME}/.local"

# WORKDIR "${HOME}"

