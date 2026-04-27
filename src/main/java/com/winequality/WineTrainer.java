package com.winequality;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import static org.apache.spark.sql.functions.col;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

public class WineTrainer {

        public static void main(String[] args) throws java.io.IOException {

                // 1. Setup Spark
                SparkSession spark = SparkSession.builder()
                                .appName("WineModelTournament")
                                .getOrCreate();

                spark.sparkContext().setLogLevel("WARN");

                // 2. Paths from args
                String trainingPath = args.length > 0 ? args[0] : "/home/ubuntu/TrainingDataset.csv";
                String validationPath = args.length > 1 ? args[1] : "/home/ubuntu/ValidationDataset.csv";
                String modelSavePath = args.length > 2 ? args[2] : "/home/ubuntu/best-wine-model";

                System.out.println("=== Wine Model Trainer ===");
                System.out.println("Training data  : " + trainingPath);
                System.out.println("Validation data: " + validationPath);
                System.out.println("Model output   : " + modelSavePath);

                // 3. Load Datasets
                Dataset<Row> trainingRaw = loadData(spark, trainingPath);
                Dataset<Row> validationRaw = loadData(spark, validationPath);

                System.out.println("Training rows  : " + trainingRaw.count());
                System.out.println("Validation rows: " + validationRaw.count());

                // Debug: print column names to verify they are clean
                System.out.println("Columns after cleaning:");
                for (String c : trainingRaw.columns()) {
                        System.out.println("  [" + c + "]");
                }

                // 4. Add class weights
                Dataset<Row> trainingWeighted = addClassWeights(trainingRaw);
                Dataset<Row> validationWeighted = addClassWeights(validationRaw);

                // 5. Feature Engineering
                Dataset<Row> trainFE = addEngineeredFeatures(trainingWeighted);
                Dataset<Row> validFE = addEngineeredFeatures(validationWeighted);

                // 6. Assemble features
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

                // 7. Fit scaler on training data only
                Dataset<Row> trainAssembled = assembler.transform(trainFE);
                Dataset<Row> validAssembled = assembler.transform(validFE);

                StandardScaler scaler = new StandardScaler()
                                .setInputCol("rawFeatures")
                                .setOutputCol("features")
                                .setWithMean(true)
                                .setWithStd(true);

                StandardScalerModel scalerModel = scaler.fit(trainAssembled);

                Dataset<Row> trainScaled = scalerModel.transform(trainAssembled)
                                .select("features", "quality", "classWeightCol");
                Dataset<Row> validScaled = scalerModel.transform(validAssembled)
                                .select("features", "quality", "classWeightCol");

                // 8. Setup Evaluator
                MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                                .setLabelCol("quality")
                                .setPredictionCol("prediction")
                                .setMetricName("f1");

                // MODEL 1: Logistic Regression
                System.out.println("\n>>> Training Model 1: Logistic Regression...");
                LogisticRegressionModel lrModel = new LogisticRegression()
                                .setLabelCol("quality")
                                .setFeaturesCol("features")
                                .setWeightCol("classWeightCol")
                                .setMaxIter(100)
                                .setRegParam(0.01)
                                .setElasticNetParam(0.0)
                                .fit(trainScaled);

                double lrF1 = evaluator.evaluate(lrModel.transform(validScaled));

                // MODEL 2: Random Forest
                System.out.println(">>> Training Model 2: Random Forest...");
                RandomForestClassificationModel rfModel = new RandomForestClassifier()
                                .setLabelCol("quality")
                                .setFeaturesCol("features")
                                .setWeightCol("classWeightCol")
                                .setNumTrees(200)
                                .setMaxDepth(20)
                                .setMaxBins(64)
                                .setMinInstancesPerNode(1)
                                .setFeatureSubsetStrategy("sqrt")
                                .setSubsamplingRate(0.8)
                                .setSeed(42)
                                .fit(trainScaled);

                double rfF1 = evaluator.evaluate(rfModel.transform(validScaled));

                // 9. Results
                System.out.println("\n========================================");
                System.out.println("      WINE MODEL TOURNAMENT RESULTS     ");
                System.out.println("========================================");
                System.out.println("Logistic Regression F1 : " + String.format("%.4f", lrF1));
                System.out.println("Random Forest F1       : " + String.format("%.4f", rfF1));
                System.out.println("========================================");

                // 10. Save Winner
                String winnerType = (rfF1 >= lrF1) ? "RF" : "LR";

                if ("RF".equals(winnerType)) {
                        System.out.println("Winner: Random Forest! Saving to: " + modelSavePath);
                        rfModel.write().overwrite().save(modelSavePath);
                } else {
                        System.out.println("Winner: Logistic Regression! Saving to: " + modelSavePath);
                        lrModel.write().overwrite().save(modelSavePath);
                }

                scalerModel.write().overwrite().save(modelSavePath + "-scaler");

                try (PrintWriter writer = new PrintWriter(modelSavePath + "-type.txt")) {
                        writer.print(winnerType);
                }

                System.out.println("Model type '" + winnerType + "' saved.");
                System.out.println("Done!");
                spark.stop();
        }

        // ── Load CSV and CLEAN column names (strips extra quotes) ─────────────────
        public static Dataset<Row> loadData(SparkSession spark, String path) {
                Dataset<Row> raw = spark.read()
                                .option("header", "true")
                                .option("inferSchema", "true")
                                .option("sep", ";")
                                .option("quote", "\"")
                                .csv(path);

                // Strip ALL extra quotes and whitespace from column names
                Dataset<Row> cleaned = raw;
                for (String colName : raw.columns()) {
                        String cleanName = colName.replaceAll("\"", "").trim();
                        if (!cleanName.equals(colName)) {
                                cleaned = cleaned.withColumnRenamed(colName, cleanName);
                        }
                }
                return cleaned;
        }

        // ── Add class weights ─────────────────────────────────────────────────────
        public static Dataset<Row> addClassWeights(Dataset<Row> df) {
                return df.withColumn("classWeightCol",
                                functions.when(col("quality").equalTo(3), 10.0)
                                                .when(col("quality").equalTo(4), 5.0)
                                                .when(col("quality").equalTo(5), 1.0)
                                                .when(col("quality").equalTo(6), 1.0)
                                                .when(col("quality").equalTo(7), 3.0)
                                                .when(col("quality").equalTo(8), 7.0)
                                                .when(col("quality").equalTo(9), 10.0)
                                                .otherwise(1.0));
        }

        // ── Add engineered features ───────────────────────────────────────────────
        public static Dataset<Row> addEngineeredFeatures(Dataset<Row> df) {
                return df
                                .withColumn("free_to_total_sulfur",
                                                col("free sulfur dioxide")
                                                                .divide(col("total sulfur dioxide").plus(0.001)))
                                .withColumn("acid_ratio",
                                                col("fixed acidity").divide(col("volatile acidity").plus(0.001)))
                                .withColumn("alcohol_density",
                                                col("alcohol").multiply(col("density")))
                                .withColumn("sulphate_alcohol",
                                                col("sulphates").multiply(col("alcohol")))
                                .withColumn("total_acid",
                                                col("fixed acidity").plus(col("volatile acidity"))
                                                                .plus(col("citric acid")));
        }
}