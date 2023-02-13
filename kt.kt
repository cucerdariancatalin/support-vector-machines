import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

// create a Spark session
val spark = SparkSession.builder().appName("SupportVectorMachineExample").getOrCreate()

// load the dataset and create a dataframe
val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("data/svm_data.csv")

// convert the target column into a categorical variable
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// assemble the feature columns into a single feature vector column
val assembler = new VectorAssembler().setInputCols(Array("feature1", "feature2", "feature3")).setOutputCol("features")
val assembledData = assembler.transform(data)

// split the data into training and testing set
val Array(trainingData, testData) = assembledData.randomSplit(Array(0.7, 0.3))

// train the classifier using LinearSVC
val svc = new LinearSVC().setMaxIter(10).setRegParam(0.1).setFeaturesCol("features").setLabelCol("indexedLabel")

// use OneVsRest to train a multi-class SVM classifier
val ovr = new OneVsRest().setClassifier(svc)
val model = ovr.fit(trainingData)

// convert the indexed labels back to original labels
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
val predictions = model.transform(testData).select("features", "indexedLabel", "prediction", "predictedLabel")

// evaluate the model using MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)

// print the accuracy of the model
println("Accuracy: " + accuracy)

// stop the Spark session
spark.stop()
