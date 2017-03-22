package com.beautypirate.ml.naivebayes

/* SimpleApp.scala */
import java.io.{BufferedWriter, File, FileWriter}

import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Sorting


/**
  * Created by azhugrov on 20.3.17.
  */
object GenerateDictionary {

  def main(args: Array[String]): Unit = {
    val nonSpamTrainFiles = args(0)
    val spamTrainFiles = args(1)
    val nonSpamTestFiles = args(2)
    val spamTestFiles = args(3)

    val conf = new SparkConf().setAppName("Naive Bayes Generate Dictionary")
    val sc = new SparkContext(conf)

    println(s"Email file pattern used => $nonSpamTrainFiles")
    val emailRDD = sc.textFile(Array(nonSpamTrainFiles, spamTrainFiles, nonSpamTestFiles, spamTestFiles).mkString(","))
    val wordCountRDD = emailRDD.flatMap(line => line.split(" ")).filter(_.length > 1).map(word => (word, 1)).reduceByKey(_ + _)
    val wordCount: Array[(String, Int)] = wordCountRDD.collect()
    Sorting.stableSort(wordCount, (first: (String, Int), second: (String, Int)) => first._2 > second._2 )
    println(s"Word count RDD => $wordCountRDD")

    saveResults(wordCount, "dictionary.txt")

    //create a tree structure from dictionary for feature vector generation


    sc.stop()
  }

  private def saveResults(wordCount: Array[(String, Int)], fileName: String): Unit = {
    val file = new File(s"/home/azhugrov/Projects/MachineLearning/NaiveBayes/data/ex6DataEmails/${fileName}")
    val bw = new BufferedWriter(new FileWriter(file))
    var index = 1
    for (pair <- wordCount.take(2500)) {
      bw.write(index+" "+pair._1+" "+pair._2+"\n")
      index += 1
    }

    bw.close()
  }

  private def directoryFiles(dirPath: String): List[String] = {
    val d = new File(dirPath)
    if (!d.exists() || !d.isDirectory) {
      throw new IllegalArgumentException(s" $dirPath should be a valid directory")
    }

    return d.listFiles().filter(_.isFile).map(_.getPath()).toList
  }

}