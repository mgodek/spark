/*
 * Author: Michal Godek
 */

package org.mgodek

import com.cloudera.sparkts.models.ARIMA
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.sql.SparkSession

/**
 * MLP classifier for MEUM project
 */
object MultilayerPerceptronClassifierMeum {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("MultilayerPerceptronClassifierMeum")
      .master("local")
      .getOrCreate()


    // The dataset is sampled from an ARIMA(1, 0, 1) model generated in R.
    val lines = scala.io.Source.fromFile("data/nn3/001.txt").getLines()
    val ts = Vectors.dense(lines.map(_.toDouble).toArray)
    val arimaModel = ARIMA.fitModel(1, 0, 1, ts)
    println("coefficients: " + arimaModel.coefficients.mkString(","))
    val forecast = arimaModel.forecast(ts, 20)
    println("forecast of next 20 observations: " + forecast.toArray.mkString(","))


    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      .load("data/nn3/001.txt")

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val model = trainer.fit(train)

    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

    spark.stop()
  }
}
