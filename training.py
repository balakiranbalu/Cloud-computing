# Import necessary libraries
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
sparkSessions1 = SparkSession.builder.appName("QualityTrainingForWine").getOrCreate()

# Schema definition for CSV data
csv_schema = StructType([
    StructField("fixed_acidity", DoubleType()),
    StructField("volatile_acidity", DoubleType()),
    StructField("citric_acid", DoubleType()),
    StructField("residual_sugar", DoubleType()),
    StructField("chlorides", DoubleType()),
    StructField("free_sulfur_dioxide", DoubleType()),
    StructField("total_sulfur_dioxide", DoubleType()),
    StructField("density", DoubleType()),
    StructField("pH", DoubleType()),
    StructField("sulphates", DoubleType()),
    StructField("alcohol", DoubleType()),
    StructField("quality", DoubleType())
])

# Reading and processing the dataset
data = spark_session.read.format("csv").schema(csv_schema).options(header=True, delimiter=';', quote='"', ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True).load('file:///home/ec2-user/WineQualityPrediction/TrainingDataset.csv')
data = data.toDF(*[c.replace('"', '') for c in data.columns])
data = data.withColumn("quality", F.when(F.col("quality") > 7, 1).otherwise(0))

# Feature vector preparation
feature_columns = dataset.columns[:-1]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
datasets = vector_assembler.transform(dataset)

# Splitting the dataset
train_data, test_data = dataset.randomSplit([0.8, 0.2])

# Model training with Random Forest Classifier
random_forest = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=200)
trained_model = random_forest.fit(train_data)

# Model prediction and evaluation
test_predictions = trained_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_metric = evaluator.evaluate(test_predictions)
print("Evaluated F1 Score: {:.4f}".format(f1_metric))

# Saving the model
trained_model_new.save("file:///home/ec2-user/WineQualityPrediction/qualitytrainingforwine")
