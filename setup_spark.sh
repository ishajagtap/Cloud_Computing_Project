#!/bin/bash
# setup_spark.sh

echo "Installing Java..."
sudo apt-get update -y
sudo apt-get install -y openjdk-11-jdk wget

echo "Downloading Spark 3.3.2..."
wget -q https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
tar -xzf spark-3.3.2-bin-hadoop3.tgz
sudo mv spark-3.3.2-bin-hadoop3 /opt/spark
rm spark-3.3.2-bin-hadoop3.tgz

echo "Configuring environment..."
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

echo "Setup complete."
