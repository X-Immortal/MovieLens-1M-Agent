FROM bde2020/hadoop-base:2.0.0-hadoop3.2.1-java8 AS hadoop

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash procps openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=hadoop /opt/hadoop-3.2.1 /opt/hadoop-3.2.1
COPY --from=hadoop /etc/hadoop /etc/hadoop
COPY --from=hadoop /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/java-8-openjdk-amd64

RUN python -m pip install --no-cache-dir streamlit==1.49.1

WORKDIR /app
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV HADOOP_HOME=/opt/hadoop-3.2.1
ENV HADOOP_CONF_DIR=/etc/hadoop
ENV PATH=/opt/hadoop-3.2.1/bin:/opt/hadoop-3.2.1/sbin:$PATH

EXPOSE 8501
