/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package weka3;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.*;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Loader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
public class Weka {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
     try{
        
        //1-load training data set  
        DataSource source = new DataSource("D:\\Weather.arff"); //create data source
        //FileReader fr = new FileReader(file); 
        //BufferedReader br =new BufferedReader(fr); 
        Instances dataSet= source.getDataSet(); //create object of type Instances to enable handling the instances
        System.out.println(dataSet); 
        System.out.println("Summary:\n"+dataSet.toSummaryString());
        
      
        //2- build a classifier tree
        dataSet.setClassIndex(dataSet.numAttributes()-1); //set the last attribute to represent classification
        J48 tree = new J48(); //create tree
        tree.buildClassifier(dataSet); //build a tree to classify the passed data set
        System.out.println(tree.graph()); //display nodes of the tree
        //System.out.println(tree.toSource(null)); //return the tree as if-then statement
        
        //3-evaluate the training set using test set
        Evaluation evaluation = new Evaluation(dataSet); //to evluate the training data set 
        DataSource testSource = new DataSource("D:\\toTest.arff"); //create test source
        Instances testSet = testSource.getDataSet(); //get the instances of the test data set
        testSet.setClassIndex(testSet.numAttributes()-1); //set the last attribute in the data set 
        evaluation.evaluateModel(tree, testSet); //(classifier to evaluate, used in avaluation)
        System.out.println("Evaluation results"+evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());
        evaluation.confusionMatrix();
        //System.out.println("After evaluation:\n"+tree.graph());
         
        // CV(dataSet);
         
         //4-prdicttion
         System.out.println("prediction:");
         System.out.println("The class attribute: "+dataSet.classAttribute());
         for(int n=0; n<dataSet.numClasses();n++){   //numClasses() number of lables, 2 in Weather dataset (yes or no)
         String possibleValue = dataSet.classAttribute().value(n); //get the possible values of the the class attribute (possible decisions)
         System.out.println("Decision "+n+" is :"+possibleValue);
         }
         
         for(int i=0;i<testSet.numInstances();i++){
         double predict = tree.classifyInstance(testSet.instance(i)); //return the index# of the class value as double (0 for yes, 1 for no)
         String Spredict = testSet.classAttribute().value((int)predict); //return the class value as string (yes,no)
         System.out.println("Instance "+i+": "+Spredict);
         }
         
         
  
     }catch(IOException E){
         E.getMessage();
     }
  }
public static void Filter(String[] option,Instances dataSet) throws Exception{

   String[] options ={"-R","1"}; //"-R" indicates remove attributr , 1 indicate the 1st attribute
        Remove remove =new Remove(); //remove object to remove selected attributes
        remove.setOptions(options);  //apply remove options
        remove.setInputFormat(dataSet); //to set the dataset that I want to apply the filter on it
        Instances newDataSet = Filter.useFilter(dataSet, remove); //store the filtered instances on a new data set
        //System.out.println(newDataSet); 
        System.out.println("Summary after filtering:\n"+newDataSet.toSummaryString()); 
}

public static void CV(Instances dataSet) throws Exception{
    
    J48 tree2 = new J48();
         Random random = new Random(1); 
         Instances randomData = new Instances(dataSet);
         randomData.randomize(random);  //create random dataset
         if(randomData.classAttribute().isNominal()){  //check if nominal(can't apply cross validation on numrical data)
             randomData.stratify(2); //divide the random data into 2 pieces
         }
         for(int x=0; x<2 ;x++){   //at each iteration use 2 folds tfor train and the rest for test
             
            Evaluation evaluation2 = new Evaluation(randomData);
            Instances train = randomData.trainCV(2, x); //(# of folds, fold number)
            Instances test = randomData.testCV(2, x);
            
            tree2.buildClassifier(train); //classify the train set
            evaluation2.evaluateModel(tree2, test); //evaluate the classifier using the test set
            System.out.println("Evaluation results"+evaluation2.toSummaryString());
            System.out.println(evaluation2.toMatrixString());
         }

}
    public static void test() throws FileNotFoundException, IOException{
       /*
        ArffSaver saver = new ArffSaver(); //create new saver
        File Newfile = new File("C:\\Users\\hyyof\\OneDrive\\سطح المكتب\\newWeather.arff");
        saver.setInstances(dataSet); //load the data set in a saver
        saver.setFile(Newfile); //set a file into the saver
        saver.writeBatch();  //write instances into the new file
    }*/
        
         /*/build a classifier nb 
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataSet);
        System.out.println(nb.getCapabilities().toString());*/
        
        /*/build a classifier SMO
        SMO smo = new SMO();
        smo.buildClassifier(dataSet);
        System.out.println(nb.getCapabilities().toString());*/
         
    
}
}

    
