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

# Spark dependencies
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
ARG spark_version="3.3.1"
ARG hadoop_version="3"
ARG scala_version
ARG spark_checksum="769db39a560a95fd88b58ed3e9e7d1e92fb68ee406689fb4d30c033cb5911e05c1942dcc70e5ec4585df84e80aabbc272b9386a208debda89522efff1335c8ff"
ARG openjdk_version="17"

ENV APACHE_SPARK_VERSION="${spark_version}" \
   HADOOP_VERSION="${hadoop_version}"

RUN apt update --yes \
   && apt install --yes --no-install-recommends \
   "openjdk-${openjdk_version}-jre-headless" \
   ca-certificates-java \
   && apt clean && rm -rf /var/lib/apt/lists/*

# Spark installation
WORKDIR /tmp

RUN if [ -z "${scala_version}" ]; then \
   wget -qO "spark.tgz" "https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"; \
   else \
   wget -qO "spark.tgz" "https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}-scala${scala_version}.tgz"; \
   fi \
   && echo "${spark_checksum} *spark.tgz" | sha512sum -c - \
   && tar xzf "spark.tgz" -C /usr/local --owner root --group root --no-same-owner \
   && rm "spark.tgz"

# Configure Spark
ENV SPARK_HOME=/usr/local/spark
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
   PATH="${PATH}:${SPARK_HOME}/bin"

RUN if [ -z "${scala_version}" ]; then \
   ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}"; \
   else \
   ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}-scala${scala_version}" "${SPARK_HOME}"; \
   fi \
   # Add a link in the before_notebook hook in order to source automatically PYTHONPATH && \
   && mkdir -p /usr/local/bin/before-notebook.d \
   && ln -s "${SPARK_HOME}/sbin/spark-config.sh" /usr/local/bin/before-notebook.d/spark-config.sh

# Configure IPython system-wide
COPY ipython_kernel_config.py "/etc/ipython/"
RUN chown -R ${NB_USER}:users /etc/ipython/

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
   'scab' \
   # pyspark
   'pyspark' \
   'findspark' 

RUN python3 -c "import findspark; findspark.init()"

# R packages including IRKernel which gets installed globally.
# r-e1071: dependency of the caret R package
RUN mamba install --quiet --yes \
   'r-base' \
   'r-caret' \
   'r-crayon' \
   'r-devtools' \
   'r-e1071' \
   'r-forecast' \
   'r-hexbin' \
   'r-htmltools' \
   'r-htmlwidgets' \
   'r-irkernel' \
   'r-nycflights13' \
   'r-randomforest' \
   'r-rcurl' \
   'r-rmarkdown' \
   'r-rodbc' \
   'r-rsqlite' \
   'r-shiny' \
   'r-tidyverse' \
   'unixodbc' && \
   mamba clean --all -f -y




# WORKDIR "${HOME}"




# Julia installation
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
# Check https://julialang.org/downloads/
ARG julia_version="1.8.5"

USER root

# install Julia packages in /opt/julia instead of ${HOME}
ENV JULIA_DEPOT_PATH=/opt/julia \
   JULIA_PKGDIR=/opt/julia \
   JULIA_VERSION="${julia_version}"

WORKDIR /tmp

RUN set -x \
   && julia_arch=$(uname -m) \
   && julia_short_arch="${julia_arch}" \
   && if [ "${julia_short_arch}" == "x86_64" ]; then \
   julia_short_arch="x64"; \
   fi; \
   julia_installer="julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" \
   && julia_major_minor=$(echo "${JULIA_VERSION}" | cut -d. -f 1,2) \
   && mkdir "/opt/julia-${JULIA_VERSION}" \
   && wget -q "https://julialang-s3.julialang.org/bin/linux/${julia_short_arch}/${julia_major_minor}/${julia_installer}" \
   && tar xzf "${julia_installer}" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1 \
   && rm "${julia_installer}" \
   && ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

# Show Julia where conda libraries are \
RUN mkdir /etc/julia \
   && echo "push!(Libdl.DL_LOAD_PATH, \"${CONDA_DIR}/lib\")" >> /etc/julia/juliarc.jl \
   # Create JULIA_PKGDIR \
   && mkdir "${JULIA_PKGDIR}" \
   && chown "${NB_USER}" "${JULIA_PKGDIR}" \
   && fix-permissions "${JULIA_PKGDIR}"



# Add Julia packages.
# Install IJulia as jovyan and then move the kernelspec out
# to the system share location. Avoids problems with runtime UID change not
# taking effect properly on the .local folder in the jovyan home dir.

USER ${NB_UID}

RUN julia -e 'import Pkg; Pkg.update()' \
   && julia -e 'import Pkg; Pkg.add("HDF5")' \
   && julia -e 'using Pkg; pkg"add IJulia"; pkg"precompile"' \
   # move kernelspec out of home \
   && mv "${HOME}/.local/share/jupyter/kernels/julia"* "${CONDA_DIR}/share/jupyter/kernels/" \
   && chmod -R go+rx "${CONDA_DIR}/share/jupyter" \
   && rm -rf "${HOME}/.local"

WORKDIR "${HOME}"

