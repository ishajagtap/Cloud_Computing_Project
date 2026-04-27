# Wine Quality Prediction – CS 643 Programming Assignment 2

## Overview

This project implements a **parallel wine quality prediction** ML application using Apache Spark MLlib on AWS EC2. It includes:

1. **Parallel model training** – RandomForestClassifier trained on 4 EC2 instances via Spark
2. **Single-machine prediction** – Load saved model, predict on test data, output F1 score
3. **Docker container** – Portable deployment of the prediction application

---

## Repository Structure

```
wine-quality-prediction/
├── src/main/java/com/winequality/
│   ├── WineModelTrainer.java      # Parallel training on 4 EC2 instances
│   └── WineQualityPredictor.java  # Single-machine prediction app
├── datasets/
│   ├── TrainingDataset.csv
│   └── ValidationDataset.csv
├── pom.xml                        # Maven build (Java 11 + Spark 3.3.2)
├── Dockerfile                     # Docker image for prediction app
└── README.md
```

---

## Prerequisites

- AWS account with EC2 and S3 access
- Java 11 (OpenJDK)
- Maven 3.6+
- Docker (for container deployment)
- AWS CLI configured (`aws configure`)

---

## Part 1 – Build the JAR

On your local machine or on an EC2 instance:

```bash
git clone <your-github-repo-url>
cd wine-quality-prediction
mvn clean package -DskipTests
```

This produces `target/wine-quality-prediction-1.0-SNAPSHOT.jar`.

---

## Part 2 – AWS Setup (4 EC2 Instances for Training)

### 2.1 Create an S3 Bucket

```bash
aws s3 mb s3://cs643-wine-bucket
aws s3 cp datasets/TrainingDataset.csv   s3://cs643-wine-bucket/
aws s3 cp datasets/ValidationDataset.csv s3://cs643-wine-bucket/
aws s3 cp target/wine-quality-prediction-1.0-SNAPSHOT.jar s3://cs643-wine-bucket/
```

### 2.2 Launch 4 EC2 Instances (Ubuntu 22.04)

In the AWS Console or CLI, launch **4 instances** (e.g., `t2.xlarge` or `m5.xlarge`):

```bash
# Example using AWS CLI (repeat 4 times or use --count 4)
aws ec2 run-instances \
  --image-id ami-0557a15b87f6559cf \   # Ubuntu 22.04 LTS (us-east-1)
  --instance-type m5.xlarge \
  --count 4 \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxxx \
  --iam-instance-profile Name=EC2_S3_Access \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spark-node}]'
```

> **Security Group** must allow:
> - SSH (port 22) from your IP
> - All TCP between the 4 instances (for Spark communication)

### 2.3 Install Java and Spark on All 4 Instances

SSH into each instance and run:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk wget

# Download Spark
wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
tar -xzf spark-3.3.2-bin-hadoop3.tgz
sudo mv spark-3.3.2-bin-hadoop3 /opt/spark

echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

### 2.4 Configure Spark Cluster

**On the Master node** (`instance-1`), edit `/opt/spark/conf/workers`:

```
<private-ip-instance-2>
<private-ip-instance-3>
<private-ip-instance-4>
```

Edit `/opt/spark/conf/spark-env.sh`:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_MASTER_HOST=<private-ip-instance-1>
```

Copy these config files to all worker nodes:

```bash
scp /opt/spark/conf/spark-env.sh ubuntu@<worker-ip>:/opt/spark/conf/
```

**Start the cluster from the master:**

```bash
/opt/spark/sbin/start-master.sh
/opt/spark/sbin/start-workers.sh spark://<master-private-ip>:7077
```

Verify at: `http://<master-public-ip>:8080`

---

## Part 3 – Parallel Model Training

Download the JAR on the master instance:

```bash
aws s3 cp s3://cs643-wine-bucket/wine-quality-prediction-1.0-SNAPSHOT.jar .
```

Run training across all 4 instances:

```bash
spark-submit \
  --class com.winequality.WineModelTrainer \
  --master spark://<master-private-ip>:7077 \
  --deploy-mode client \
  --num-executors 4 \
  --executor-cores 2 \
  --executor-memory 4g \
  --driver-memory 4g \
  wine-quality-prediction-1.0-SNAPSHOT.jar \
  s3://cs643-wine-bucket/TrainingDataset.csv \
  s3://cs643-wine-bucket/ValidationDataset.csv \
  s3://cs643-wine-bucket/wine-model
```

> The `--num-executors 4` ensures Spark distributes the training across all 4 EC2 instances.
>
> The model is saved to `s3://cs643-wine-bucket/wine-model`.

---

## Part 4 – Single-Machine Prediction (Without Docker)

SSH into **one** EC2 instance. Download the model and test data:

```bash
aws s3 cp s3://cs643-wine-bucket/wine-quality-prediction-1.0-SNAPSHOT.jar .
aws s3 sync s3://cs643-wine-bucket/wine-model ./wine-model
aws s3 cp s3://cs643-wine-bucket/TestDataset.csv .   # or use ValidationDataset.csv
```

Run prediction:

```bash
spark-submit \
  --class com.winequality.WineQualityPredictor \
  --master local[*] \
  --driver-memory 4g \
  wine-quality-prediction-1.0-SNAPSHOT.jar \
  TestDataset.csv \
  ./wine-model
```

The F1 score is printed to stdout.

---

## Part 5 – Docker Container (Prediction App)

### 5.1 Build the Docker Image

On a machine with Docker installed:

```bash
# Make sure JAR is built first
mvn clean package -DskipTests

# Build Docker image
docker build -t wine-quality-prediction .
```

### 5.2 Test the Container Locally

```bash
# Mount your local data directory into the container
docker run --rm \
  -v $(pwd)/datasets:/data \
  -v $(pwd)/wine-model:/app/wine-model \
  wine-quality-prediction \
  /data/ValidationDataset.csv /app/wine-model
```

### 5.3 Push to Docker Hub

```bash
docker login

# Tag with your Docker Hub username
docker tag wine-quality-prediction <your-dockerhub-username>/wine-quality-prediction:latest

# Push
docker push <your-dockerhub-username>/wine-quality-prediction:latest
```

### 5.4 Pull and Run from Docker Hub (on EC2)

On any EC2 instance with Docker installed:

```bash
# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker

# Pull image
docker pull <your-dockerhub-username>/wine-quality-prediction:latest

# Download model and test data
aws s3 sync s3://cs643-wine-bucket/wine-model ./wine-model
aws s3 cp s3://cs643-wine-bucket/TestDataset.csv ./TestDataset.csv

# Run prediction
docker run --rm \
  -v $(pwd):/data \
  <your-dockerhub-username>/wine-quality-prediction:latest \
  /data/TestDataset.csv /data/wine-model
```

---

## Model Details

| Parameter         | Value                          |
|-------------------|--------------------------------|
| Algorithm         | RandomForestClassifier         |
| Feature columns   | 11 (all except quality)        |
| Target column     | quality (classes 3–8)          |
| Hyperparameter CV | numTrees: [50,100,150], maxDepth: [8,10,12] |
| Cross-validation  | 3-fold, parallelism=4          |
| Metric            | F1 Score (weighted)            |
| Spark version     | 3.3.2                          |
| Java version      | 11                             |

---

## Links

- **GitHub**: `https://github.com/<your-username>/wine-quality-prediction`
- **Docker Hub**: `https://hub.docker.com/r/<your-dockerhub-username>/wine-quality-prediction`

---

## Grading Checklist

| Component                                  | Points | Status |
|--------------------------------------------|--------|--------|
| Parallel training on 4 EC2 instances       | 50     | ✅     |
| Single-machine prediction application      | 25     | ✅     |
| Docker container for prediction app        | 25     | ✅     |
| **Total**                                  | **100**| ✅     |