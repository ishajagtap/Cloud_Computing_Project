package com.winequality;

import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.functions;
import static org.apache.spark.sql.functions.col;

import java.nio.file.Files;
import java.nio.file.Paths;

public class WineQualityPredictor {

    public static void main(String[] args) throws java.io.IOException {

        // Single machine
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPredictionApp")
                .master("local[*]")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        String testDataPath = args.length > 0 ? args[0] : "/home/ubuntu/ValidationDataset.csv";
        String modelPath = args.length > 1 ? args[1] : "/home/ubuntu/best-wine-model";
        String scalerPath = modelPath + "-scaler";
        String modelTypePath = modelPath + "-type.txt";

        System.out.println("=== Wine Quality Predictor ===");
        System.out.println("Test data : " + testDataPath);
        System.out.println("Model path: " + modelPath);

        // Load and clean test data
        Dataset<Row> rawData = loadData(spark, testDataPath);
        System.out.println("Test rows : " + rawData.count());

        // Feature engineering (must match trainer exactly)
        Dataset<Row> feData = addEngineeredFeatures(rawData);

        // Assemble features
        String[] featureCols = {
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                "density", "pH", "sulphates", "alcohol",
                "free_to_total_sulfur", "acid_ratio",
                "alcohol_density", "sulphate_alcohol", "total_acid"
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("rawFeatures");

        Dataset<Row> assembled = assembler.transform(feData);

        // Load scaler
        System.out.println("Loading scaler from: " + scalerPath);
        StandardScalerModel scalerModel = StandardScalerModel.load(scalerPath);
        Dataset<Row> scaledData = scalerModel.transform(assembled);

        // Read model type
        String modelType = new String(Files.readAllBytes(Paths.get(modelTypePath))).trim();
        System.out.println("Model type: " + modelType);

        // Load correct model type
        Dataset<Row> predictions;
        if ("RF".equals(modelType)) {
            System.out.println("Loading Random Forest model from: " + modelPath);
            RandomForestClassificationModel model = RandomForestClassificationModel.load(modelPath);
            predictions = model.transform(scaledData);
        } else {
            System.out.println("Loading Logistic Regression model from: " + modelPath);
            LogisticRegressionModel model = LogisticRegressionModel.load(modelPath);
            predictions = model.transform(scaledData);
        }

        // Show results
        System.out.println("\n--- PREDICTION RESULTS (First 20 Rows) ---");
        predictions.select("quality", "prediction").show(20);

        // F1 Score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);

        MulticlassClassificationEvaluator accEval = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = accEval.evaluate(predictions);

        System.out.println("\n========================================");
        System.out.println("  Final Test F1 Score : " + String.format("%.4f", f1));
        System.out.println("  Accuracy            : " + String.format("%.4f", accuracy));
        System.out.println("========================================");

        System.out.println("\nPrediction distribution:");
        predictions.groupBy("prediction").count().orderBy("prediction").show();

        spark.stop();
    }

    // Load CSV and clean column names
    public static Dataset<Row> loadData(SparkSession spark, String path) {
        Dataset<Row> raw = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("sep", ";")
                .option("quote", "\"")
                .csv(path);

        Dataset<Row> cleaned = raw;
        for (String colName : raw.columns()) {
            String cleanName = colName.replaceAll("\"", "").trim();
            if (!cleanName.equals(colName)) {
                cleaned = cleaned.withColumnRenamed(colName, cleanName);
            }
        }
        return cleaned;
    }

    // Add engineered features (must match WineTrainer exactly)
    public static Dataset<Row> addEngineeredFeatures(Dataset<Row> df) {
        return df
                .withColumn("free_to_total_sulfur",
                        col("free sulfur dioxide").divide(col("total sulfur dioxide").plus(0.001)))
                .withColumn("acid_ratio",
                        col("fixed acidity").divide(col("volatile acidity").plus(0.001)))
                .withColumn("alcohol_density",
                        col("alcohol").multiply(col("density")))
                .withColumn("sulphate_alcohol",
                        col("sulphates").multiply(col("alcohol")))
                .withColumn("total_acid",
                        col("fixed acidity").plus(col("volatile acidity")).plus(col("citric acid")));
    }
}