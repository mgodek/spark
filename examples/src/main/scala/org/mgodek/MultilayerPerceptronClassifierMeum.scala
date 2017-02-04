/*
 * Author: Michal Godek
 */

package org.mgodek

import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * MLP classifier for MEUM project
 */
object MultilayerPerceptronClassifierMeum {

  def lagMatTrimBoth(x: Array[Double], maxLag: Int, includeOriginal: Boolean)
    : Array[Array[Double]] = {
    val numObservations = x.length
    val numRows = numObservations - maxLag
    val numCols = maxLag + (if (includeOriginal) 1 else 0)
    val lagMat = Array.ofDim[Double](numRows, numCols)

    val initialLag = if (includeOriginal) 0 else 1

    for (r <- 0 until numRows) {
      for (c <- initialLag to maxLag) {
        lagMat(r)(c - initialLag) = x(r + maxLag - c)
      }
    }
    lagMat
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("MultilayerPerceptronClassifierMeum")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    val data = spark.read.text("data/nn3/001.txt")
    val df = data.withColumnRenamed("value", "features")
    df.show()
    df.printSchema
    val toDouble = udf[Double, String]( _.toDouble)
    val featureDf = df.withColumn("features", toDouble(df("features")))
    featureDf.show()
    featureDf.printSchema
    val matSeries = lagMatTrimBoth(featureDf.select("features").map(r => r(0).asInstanceOf[Double]).collect(), 7, false)
    print(matSeries.map(_.mkString).mkString("\n"))

    // Split the data into train and test
    val splits = featureDf.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 1
    val layers = Array[Int](4, 5, 4, 1)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(1000)

    // train the model
    val model = trainer.fit(matSeries)

    // compute accuracy on the test set
    val result = model.transform(test)
    println("result: " + result)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

    spark.stop()
  }
}
