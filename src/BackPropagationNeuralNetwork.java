
/**
 * File: BackPropagationNeuralNetwork.java
 * 
 * FeedForward and Backpropagation Neural Network implementation. Provides implementation for 
 * feed forward, error calculation and back propagation
 * 
 * @author Binit
 *
 */
public class BackPropagationNeuralNetwork {

	// Total error of the training
	public float overallError;
	// number of input neurons
	public int noOfInputNeurons;
	// number of hidden neurons
	public int noOfHiddenNeurons;
	// number of output neurons
	public int noOfOutputNeurons;
	// total number of neurons in neural network
	public int totalNeurons;
	// total number of weights required in the neural network
	public int totalWeightsReq;
	// learning rate
	public float learnRate;
	// output from each neurons
	public float feedOut[];
	// weights for each link
	public float weight[];
	// previous error
	public float error[];
	// weight difference for each link
	public float weightDelta[];
	// changed weights for each link
	public float weightChange[];
	// momentum for training
	public float momentum;
	// change in error
	public float errorDelta[];


	/**
	 * initialize the neural network
	 *
	 * @param noOfInputNodes number of input neurons
	 * @param noOfHiddenNodes number of hidden neurons
	 * @param noOfOutputNodes number of output neurons
	 * @param learnRate learning rate for training
	 * @param momentum momentum for training
	 */
	public BackPropagationNeuralNetwork(int noOfInputNodes, int noOfHiddenNodes, int noOfOutputNodes, float learnRate, float momentum) {

		this.noOfInputNeurons = noOfInputNodes;
		this.noOfHiddenNeurons = noOfHiddenNodes;
		this.noOfOutputNeurons = noOfOutputNodes;
		this.learnRate = learnRate;
		this.momentum = momentum;
		this.totalNeurons = noOfInputNodes + noOfHiddenNodes + noOfOutputNodes;
		this.totalWeightsReq = (noOfInputNodes * noOfHiddenNodes) + (noOfHiddenNodes * noOfOutputNodes);

		this.feedOut    = new float[this.totalNeurons];
		this.weight   = new float[this.totalWeightsReq];
		this.weightChange = new float[this.totalWeightsReq];
		this.errorDelta = new float[this.totalNeurons];
		this.error = new float[this.totalNeurons];
		this.weightDelta = new float[this.totalWeightsReq];

		this.init();
	}


	/**
	 * gives output for any data given as input to neural network
	 *
	 * @param input	input to be tested, feed to neural network
	 * @return	output from neural network
	 */
	public float[] feedForward(float input[]) {
		float output[] = new float[this.noOfOutputNeurons];
		final int hiddenNeuronIndex = this.noOfInputNeurons;
		final int outputNeuronIndex = this.noOfInputNeurons + this.noOfHiddenNeurons;

		// initialize output from input neurons with the input values
		for (int i = 0; i < this.noOfInputNeurons; i++) {
			this.feedOut[i] = input[i];
		}

		// output from hidden neurons
		int index = 0;
		for (int i = hiddenNeuronIndex; i < outputNeuronIndex; i++) {
			float sum = 0;

			for (int j = 0; j < this.noOfInputNeurons; j++) {
				sum += feedOut[j] * this.weight[index++];
			}
			this.feedOut[i] = sigmoid(sum);
		}

		// output from output neurons
		for (int i = outputNeuronIndex; i < this.totalNeurons; i++) {
			float sum = 0;

			for (int j = hiddenNeuronIndex; j < outputNeuronIndex; j++) {
				sum += this.feedOut[j] * this.weight[index++];
			}
			this.feedOut[i] = sigmoid(sum);
			output[i - outputNeuronIndex] = this.feedOut[i];
		}
		return output;
	} // feedForward


	/**
	 * compute the error between actual and expected output
	 *
	 * @param expected output expected from output neurons
	 */
	public void calculateError(float[] expected) {
		final int hiddenNeuronIndex = this.noOfInputNeurons;
		final int outputNeuronIndex = this.noOfInputNeurons + this.noOfHiddenNeurons;

		// reset all the hidden and output layer errors
		for (int i = this.noOfInputNeurons; i < this.totalNeurons; i++) {
			this.error[i] = 0;
		}

		// compute the errors at output layer
		for (int i = outputNeuronIndex; i < this.totalNeurons; i++) {
			this.error[i] = expected[i - outputNeuronIndex] - this.feedOut[i];
			this.overallError += this.error[i] * this.error[i];
			this.errorDelta[i] = this.error[i] * this.feedOut[i] * (1 - this.feedOut[i]);
		}

		// compute the errors at hidden layer
		int index = this.noOfInputNeurons * this.noOfHiddenNeurons;
		for (int i = outputNeuronIndex; i < this.totalNeurons; i++) {
			for (int j = hiddenNeuronIndex; j < outputNeuronIndex; j++) {
				this.weightDelta[index] += this.errorDelta[i] * this.feedOut[j];
				this.error[j] += this.weight[index++] * this.errorDelta[i];
			}
		}

		// compute the error deltas at output layer
		for (int i = hiddenNeuronIndex; i < outputNeuronIndex; i++) {
			this.errorDelta[i] = this.error[i] * this.feedOut[i] * (1 - this.feedOut[i]);
		}

		// compute the error deltas at hidden layer
		index = 0;
		for (int i = hiddenNeuronIndex; i < outputNeuronIndex; i++) {
			for (int j = 0; j < hiddenNeuronIndex; j++) {
				this.weightDelta[index] += this.errorDelta[i] * this.feedOut[j];
				this.error[j] += this.weight[index++] * this.errorDelta[i];
			}
		}
	} // calculateError

	
	/**
	 * backpropagate errors at each layer to update weights at each layer
	 */
	public void backPropagate() {
		// update weights of each link
		for (int i = 0; i < this.weight.length; i++) {
			this.weightChange[i] = (this.learnRate * this.weightDelta[i]) + (this.momentum * this.weightChange[i]);
			this.weight[i] += this.weightChange[i];
			this.weightDelta[i] = 0;
		}
	} // backPropagate
	

	/**
	 * initialize the weights
	 */
	public void init() {
		// reset weights
		for (int i = 0; i < this.weight.length; i++) {
			this.weight[i] = (float)(0.5 - (Math.random()));
			this.weightChange[i] = 0;
			this.weightDelta[i] = 0;
		}
	} // init
	
	
	/**
	 * gives the mean squared error
	 *
	 * @param sampleLen total number of training samples
	 * @return error of neural network
	 */
	public float getError(int sampleLen) {
		float error = (float)Math.sqrt(this.overallError / (sampleLen * this.noOfOutputNeurons));
		this.overallError = 0;
		return error;
	} // getError


	/**
	 * Calculate the sigmoid activation function
	 *
	 * @param inputSum sum of all the links to the neuron
	 * @return result of activation function
	 */
	public float sigmoid(float inputSum) {
		return (float)(1.0 / (1 + (float)Math.exp(-1.0 * inputSum)));
	} // sigmoid
} // class BackPropagationNeuralNetwork