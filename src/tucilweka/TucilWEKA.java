/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package tucilweka;

/**
 * @author Rakhmatullah Yoga Sutrisna - 13512053
 *         Hendro Triokta Brianto - 13512081
 */

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;

public class TucilWEKA {
    private static Instances dataset;
    private static Classifier clasifier;
    private static Evaluation eval;
    private static String[] args;
    
    static void ReadDataset(String FilePath) throws Exception {
        dataset = DataSource.read(FilePath);
        dataset.setClassIndex(dataset.numAttributes()-1);
    }
    /* J48 */
    static void TenFoldTrain_J48() throws Exception {
        eval = new Evaluation(dataset);
        J48 tree = new J48();
        eval.crossValidateModel(tree, dataset, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    static void FullTraining_J48() throws Exception {
        Classifier cls = new J48();
        cls.buildClassifier(dataset);
        eval = new Evaluation(dataset);
        eval.evaluateModel(cls, dataset);
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    
    /* Naive Bayes */
    static void TenFoldTrain_NaiveBayes() throws Exception {
        eval = new Evaluation(dataset);
        NaiveBayes tree = new NaiveBayes();
        eval.crossValidateModel(tree, dataset, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    static void FullTraining_NaiveBayes() throws Exception {
        Classifier cls = new NaiveBayes();
        cls.buildClassifier(dataset);
        eval = new Evaluation(dataset);
        eval.evaluateModel(cls, dataset);
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    
    /* IBK */
    static void TenFoldTrain_IBk() throws Exception {
        eval = new Evaluation(dataset);
        IBk tree = new IBk();
        eval.crossValidateModel(tree, dataset, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    static void FullTraining_IBk() throws Exception {
        Classifier cls = new IBk();
        cls.buildClassifier(dataset);
        eval = new Evaluation(dataset);
        eval.evaluateModel(cls, dataset);
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    
    /* Multilayer Perceptron */
    static void TenFoldTrain_aNN() throws Exception {
        eval = new Evaluation(dataset);
        MultilayerPerceptron tree = new MultilayerPerceptron();
        eval.crossValidateModel(tree, dataset, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    static void FullTraining_aNN() throws Exception {
        Classifier cls = new MultilayerPerceptron();
        cls.buildClassifier(dataset);
        eval = new Evaluation(dataset);
        eval.evaluateModel(cls, dataset);
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    
    static void SaveModel(Classifier cls) throws Exception {
        clasifier = cls;
        clasifier.buildClassifier(dataset);
        SerializationHelper.write(cls.getClass().getSimpleName()+".model", cls);
    }
    static void ReadModel(String Filepath) throws Exception {
        clasifier = (Classifier) SerializationHelper.read(Filepath);
    }
    static void Classify(String FilePath) throws Exception {
        Instances unlabeled = DataSource.read(FilePath);
        unlabeled.setClassIndex(unlabeled.numAttributes()-1);
        Instances labeled = new Instances(unlabeled);
        for(int i=0; i<unlabeled.numInstances(); i++) {
            double clsLabel = clasifier.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        DataSink.write("newLabeled.arff", labeled);

    }
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        // membaca dataset awal
        ReadDataset("weather.nominal.arff");
        
        // membuat model dan menyimpannya
        SaveModel(new J48());
        SaveModel(new NaiveBayes());
        SaveModel(new IBk());
        SaveModel(new MultilayerPerceptron());
        
        // melakukan training dengan 10-fold dan full-training
        /* J48 */
        TenFoldTrain_J48();
        FullTraining_J48();
        
        /* Naive Bayes */
        TenFoldTrain_NaiveBayes();
        FullTraining_NaiveBayes();
        
        /* IBk (kNN) */
        TenFoldTrain_IBk();
        FullTraining_IBk();
        
        /* aNN */
        TenFoldTrain_aNN();
        FullTraining_aNN();
        
        // membaca model yang telah disimpan pada file eksternal
        ReadModel("J48.model");
        ReadModel("NaiveBayes.model");
        ReadModel("IBk.model");
        ReadModel("MultilayerPerceptron.model");
        
        // melakukan klasifikasi terhadap suatu dataset yang belum terlabel berdasarkan model yang telah diload
        Classify("weather.nominal-unlabeled.arff");
        
    }
    
}
