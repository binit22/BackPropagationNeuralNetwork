import java.io.FileNotFoundException;

/**
 * File: ClassifyLanguage.java
 * 
 * Takes the input samples and calls methods of BackPropagationNeuralNetwork to train 
 * the Neural Network and applies test data to test the working of Neural Network.
 * 
 * @author Binit
 *
 */
public class ClassifyLanguage {
	
	// learning rate
	public float learnRate = 0.9f;
	// minimum error tolerable
	public float minError = 0.001f;
	// difference in output result allowed 
	public float epsilon = 0.1f;
	// maximum number of iterations
	public long maxIteration = 50000;
	// momentum allowed
	public float momentum = 0.5f;
	
	// number of input neurons
	public int noOfInputNodes = Features.english.length + Features.italian.length + Features.dutch.length;
	// number of hidden neurons
	public int noOfHiddenNodes = noOfInputNodes/2;
	// number of output neurons
	public int noOfOutputNodes = 3;
	
	// number of input samples
	public int inputSamples = 9;
	// 9 input sample
	public float[][] input = new float[inputSamples][noOfInputNodes];
	public float[][] expectedOut = new float[inputSamples][noOfOutputNodes];

	// errors after each epoch
	public float[] error = new float[(int)maxIteration];
	
	/**
	 * Trains and tests neural network
	 * 
	 * @param fileName	name of the file to test neural network
	 * @throws FileNotFoundException
	 */
	public void test(String fileName) throws FileNotFoundException, Exception{
		
		System.out.println("Training:");

		Neuron neuron = new Neuron(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes);
		
		// get the inputs and expected outputs for each input sample
		input[0] = neuron.getInput("english1", "txt");
		expectedOut[0] = neuron.getExpectedOutput("english");
		neuron = new Neuron(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes);
		input[1] = neuron.getInput("italian1", "txt");
		expectedOut[1] = neuron.getExpectedOutput("italian");
		neuron = new Neuron(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes);
		input[2] = neuron.getInput("dutch1", "txt");
		expectedOut[2] = neuron.getExpectedOutput("dutch");

		input[3] = neuron.getInput("english2", "txt");
		expectedOut[3] = neuron.getExpectedOutput("english");
		neuron = new Neuron(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes);
		input[4] = neuron.getInput("italian2", "txt");
		expectedOut[4] = neuron.getExpectedOutput("italian");
		neuron = new Neuron(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes);
		input[5] = neuron.getInput("dutch2", "txt");
		expectedOut[5] = neuron.getExpectedOutput("dutch");

		input[6] = neuron.getInput("english3", "txt");
		expectedOut[6] = neuron.getExpectedOutput("english");
		neuron = new Neuron(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes);
		input[7] = neuron.getInput("italian3", "txt");
		expectedOut[7] = neuron.getExpectedOutput("italian");
		neuron = new Neuron(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes);
		input[8] = neuron.getInput("dutch3", "txt");
		expectedOut[8] = neuron.getExpectedOutput("dutch");

		BackPropagationNeuralNetwork network = new BackPropagationNeuralNetwork(noOfInputNodes, noOfHiddenNodes, noOfOutputNodes, learnRate, momentum);

		int i = 0; float error = 1;
		for (; i < maxIteration && error > minError; i++) {
			for (int j = 0; j < input.length; j++) {
				// calculate the actual output
				network.feedForward(input[j]);
				// calculate error between actual and expected output
				network.calculateError(expectedOut[j]);
				// back propagate the error to train the network
				network.backPropagate();
			}
			error = network.getError(input.length);
			this.error[i] = error;
			// System.out.println( "Epoch #" + i + ", Error: " + error);
		}
		
		// Plot the error
//		for (i = 0; i < maxIteration; i++) {
//			System.out.print(this.error[i] + " ");
//		}System.out.println();
		// 

		System.out.println("Sum of squared errors = " + error);
		System.out.println("EPOCH " + i+"\n");

		System.out.println("TEST:");
		String file[] = fileName.split("\\.");
		
		// calculate the output for test data
		float out[] = network.feedForward(neuron.getInput(file[0], file[1]));
		
		// output values of each output neurons
		System.out.println("Output from Neuron 1(English): " + out[0]);
		System.out.println("Output from Neuron 2(Italian): " + out[1]);
		System.out.println("Output from Neuron 3(Dutch): " + out[2]);

		// classify the language
		float expected = 1.0f;
		if((expected - out[0] - epsilon) < (minError))
			System.out.println("Language is English");
		else if((expected - out[1] - epsilon) < (minError))
			System.out.println("Language is Italian");
		else if((expected - out[2] - epsilon) < (minError))
			System.out.println("Language is Dutch");
		else{
			if(out[0] > out[1] && out[0] > out[2])
				System.out.println("Language is English");
			else if(out[1] > out[0] && out[1] > out[2])
				System.out.println("Language is Italian");
			else if(out[2] > out[0] && out[2] > out[1])
				System.out.println("Language is Dutch");
			else
				System.out.println("Cannot identify");
		}
	} // test

	
	/**
	 * @param args
	 */
	public static void main(String args[])
	{
		try{
			new ClassifyLanguage().test(args[0]);
		} catch(Exception ex){
			ex.printStackTrace();
		}
	} // main
} // ClassifyLanguage