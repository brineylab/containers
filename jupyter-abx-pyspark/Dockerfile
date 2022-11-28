ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/jupyter-abx
FROM $BASE_IMG

# This image is modified from images initially created by the Kubeflow team:
# https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers
# and the Jupyter team:
# https://github.com/jupyter/docker-stacks
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

USER root

# set shell to bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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

RUN apt-get update --yes \
  && apt-get install --yes --no-install-recommends \
  "openjdk-${openjdk_version}-jre-headless" \
  ca-certificates-java \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

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

# copy requirements.txt
COPY requirements.txt /tmp
RUN chown -R ${NB_USER}:users /tmp/requirements.txt

USER ${NB_UID}

# install pyspark and make available in jupyterlab
RUN python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
  && rm -f /tmp/requirements.txt
RUN python3 -c "import findspark; findspark.init()"

WORKDIR "${HOME}"

EXPOSE 4040


