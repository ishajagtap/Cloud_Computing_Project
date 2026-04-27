# ─────────────────────────────────────────────────────────────────────────────
# Wine Quality Prediction – Docker Image
# Base: Ubuntu 22.04 + Java 11 + Spark 3.3.2
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    wget \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ── Spark 3.3.2 ───────────────────────────────────────────────────────────────
ENV SPARK_VERSION=3.3.2
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"

RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && tar -xzf "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && mv "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}" \
    && rm "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

# ── Java environment ──────────────────────────────────────────────────────────
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# ── Application ───────────────────────────────────────────────────────────────
WORKDIR /app

# Copy the fat JAR built by Maven
COPY target/wine-quality-prediction-1.0-SNAPSHOT.jar /app/wine-quality-prediction.jar

# Copy datasets into image (optional; can also mount via -v flag at runtime)
COPY datasets/ /app/datasets/

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Default: run predictor. Override CMD for training.
# docker run wine-quality /data/TestDataset.csv /data/wine-model
ENTRYPOINT ["spark-submit", \
    "--class", "com.winequality.WineQualityPredictor", \
    "--master", "local[*]", \
    "--driver-memory", "4g", \
    "/app/wine-quality-prediction.jar"]

# Default arguments (can be overridden at docker run time)
CMD ["/app/datasets/ValidationDataset.csv", "/app/wine-model"]