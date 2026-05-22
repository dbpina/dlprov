FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV NEO4J_HOME=/var/lib/neo4j

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    openjdk-11-jdk \
    wget \
    curl \
    gnupg2 \
    ca-certificates \
    lsb-release \
    apt-transport-https \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --upgrade pip

# MonetDB
RUN wget -qO - https://www.monetdb.org/downloads/MonetDB-GPG-KEY | gpg --dearmor -o /usr/share/keyrings/monetdb.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/monetdb.gpg] https://dev.monetdb.org/downloads/deb/ $(lsb_release -sc) monetdb" > /etc/apt/sources.list.d/monetdb.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    monetdbd \
    monetdb-client \
    && rm -rf /var/lib/apt/lists/*

# Neo4j
RUN wget -qO - https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor -o /usr/share/keyrings/neo4j.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 4.1" > /etc/apt/sources.list.d/neo4j.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    neo4j=1:4.1.0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    tensorflow==2.15.0 \
    torch==2.2.0 \
    torchvision==0.17.0 \
    scikit-learn==1.4.0 \
    numpy==1.24.3 \
    Pillow==10.2.0 \
    matplotlib==3.8.3 \
    pymonetdb \
    neo4j \
    prov \
    pydot \
    provdbconnector

WORKDIR /opt/dlprov

COPY DfAnalyzer/ ./DfAnalyzer/
COPY lib-python/ ./lib-python/
COPY generate-prov/ ./generate-prov/
COPY Example/ ./Example/
COPY run_experiment.sh .
COPY run_df_experiment.sh .

RUN pip3 install --no-cache-dir ./lib-python/

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 50000 7474 7687 22000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
