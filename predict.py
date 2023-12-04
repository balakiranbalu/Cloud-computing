# Import required modules
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from werkzeug.utils import secure_filename
import pyspark.sql.functions as F

# Initialize Flask and CORS
app_new = Flask(__name__)
CORS(app_new)

# Initialize Spark session for inference
sparkSessions1 = SparkSession.builder.appName("QualityInferenceForWine").getOrCreate()
rfMode = RandomForestClassificationModel.load("/src/qualitytrainingforwine")

# Define schema for incoming CSV data
DataSchema = StructType([
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

@app_new.route("/predict", methods=["POST"])
def prediction():
    # Receive and save the uploaded file
    new_file = request.files["file"]
    secure_name = secure_filename(new_file.filename)
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, secure_name)
    new_file_file.save(path)

    # Process the dataset
    valid_dataset = sparkSessions1.read.format("csv").schema(DataSchema).options(header=True, delimiter=';', quote='"').load(path)
    valid_dataset = valid_dataset.withColumn("quality", F.when(F.col("quality") > 7, 1).otherwise(0))
    
    # Feature vectorization
    features = VectorAssembler(inputCols=validation_dataset.columns[:-1], outputCol="features")
    valid_dataset = features.transform(valid_dataset)

    # Predict using the model
    prediction = rf_model.transform(valid_dataset)
    
    # Model evaluation
    evaluation = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_metric = evaluation.evaluate(prediction_results)

    # Return JSON response
    return jsonify({"f1_score": f1_metric})

if __name__ == "__main__":
    app_new_1.run(host="0.0.0.0", port=5000)

