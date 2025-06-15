
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder.appName("BioinformaticsML").getOrCreate()

# Load the dataset
# Assuming leukemia_expression.csv is accessible within the Spark environment
df = spark.read.csv("leukemia_expression.csv", header=True, inferSchema=True)

# Prepare features vector
feature_cols = df.columns[:-1]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df)

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2])

# Train a RandomForestClassifier model
rf = RandomForestClassifier(labelCol=df.columns[-1], featuresCol="features", numTrees=100)
model = rf.fit(train_data)

# Make predictions and evaluate the model
predictions = model.transform(test_data)
accuracy = predictions.filter(predictions[df.columns[-1]] == predictions.prediction).count() / test_data.count()

# Print the accuracy
print("Accuracy on test data:", accuracy)

# Stop Spark Session
spark.stop()


