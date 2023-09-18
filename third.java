// J48 is a machine learning decision tree classification algorithm based on Iterative Dichotomiser 3.
// It is very helpful in examine the data categorically and continuously.
// To build our J48 machine learning model we’ll use the weka tool.

// Java Program for Creating a Model Based on J48 Classifier

// Importing required classes
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

// Main class
public class BreastCancer {

	// Main driver method
	public static void main(String args[])
	{

		// Try block to check for exceptions
		try {

			// Creating J48 classifier
			J48 j48Classifier = new J48();

			// Dataset path
			String breastCancerDataset
				= "/home/droid/Tools/weka-3-8-5/data/breast-cancer.arff";

			// Create bufferedreader to read the dataset
			BufferedReader bufferedReader
				= new BufferedReader(
					new FileReader(breastCancerDataset));

			// Create dataset instances
			Instances datasetInstances
				= new Instances(bufferedReader);

			// Set Target Class
			datasetInstances.setClassIndex(
				datasetInstances.numAttributes() - 1);

			// Evaluation
			Evaluation evaluation
				= new Evaluation(datasetInstances);

			// Cross Validate Model with 10 folds
			evaluation.crossValidateModel(
				j48Classifier, datasetInstances, 10,
				new Random(1));
			System.out.println(evaluation.toSummaryString(
				"\nResults", false));
		}

		// Catch block to check for rexceptions
		catch (Exception e) {

			// Print and display the display message
			// using getMessage() method
			System.out.println("Error Occurred!!!! \n"
							+ e.getMessage());
		}

		// Display message to be printed ion console
		// when program is successfully executed
		System.out.print("Successfully executed.");
	}
}



