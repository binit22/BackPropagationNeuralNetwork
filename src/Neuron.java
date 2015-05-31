import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


/**
 * File	Neuron.java
 * @author Binit
 *
 * Provides input values to the neural network and gives the expected output for each language
 */
public class Neuron {

	public static String english = "english";
	public static String italian = "italian";
	public static String dutch = "dutch";
	
	// total work count in the sample input text
	public int wordCount;
	// count of each features from the given text
	public int engCount[];
	public int itaCount[];
	public int dutCount[];

	// input values to neural network
	public float input[];
	// expected output values for each language
	public float expectedOutput[];

	public Neuron(int noOfInputNodes, int noOfHiddenNodes, int noOfOutputNodes){
		this.input = new float[noOfInputNodes];
		this.expectedOutput = new float[noOfOutputNodes];
	}

	/**
	 * reset all the word counts for another sample
	 */
	public void reset(){
		this.wordCount = 0;
		this.engCount = new int[Features.english.length];
		this.itaCount = new int[Features.italian.length];
		this.dutCount = new int[Features.dutch.length];
	} // reset
	
	
	/**
	 * @param language	name of the language for which expected output is desired
	 * @return	values of expected output for the given language
	 */
	public float[] getExpectedOutput(String language){
		
		if(english.equals(language))
			this.expectedOutput[0] = 1;
		else
			this.expectedOutput[0] = 0;
		
		if(italian.equals(language))
			this.expectedOutput[1] = 1;
		else
			this.expectedOutput[1] = 0;
		
		if(dutch.equals(language))
			this.expectedOutput[2] = 1;
		else
			this.expectedOutput[2] = 0;

		return this.expectedOutput;
	} // getOutput
	
	
	/**
	 * @param language	name of the input language for which input values are required
	 * @param fileExt	file extension
	 * @return	input values to the neural network
	 * @throws FileNotFoundException
	 * @throws Exception
	 */
	public float[] getInput(String language, String fileExt) throws FileNotFoundException, Exception{
		reset();
		
		Scanner sc = new Scanner(new File(language+"."+fileExt));
		String text = "";

		// until all the text from sample input is considered
		while(sc.hasNext()){
			wordCount++;
			text = sc.next().trim();
			for(int i = 0; i < Features.english.length; i++){
				if(text.contains(Features.english[i]) || text.toLowerCase().contains(Features.english[i])){
					this.engCount[i]++;
				}
			}
			for(int i = 0; i < Features.italian.length; i++){
				if(text.contains(Features.italian[i]) || text.toLowerCase().contains(Features.italian[i])){
					this.itaCount[i]++;
				}
			}
			for(int i = 0; i < Features.dutch.length; i++){
				if(text.contains(Features.dutch[i]) || text.toLowerCase().contains(Features.dutch[i])){
					this.dutCount[i]++;
				}
			}
		}

		int index = 0;
		for(int i = 0; i < Features.english.length; i++){
			this.input[index++] = (float)this.engCount[i] / this.wordCount;
		}
		for(int i = 0; i < Features.italian.length; i++){
			this.input[index++] = (float)this.engCount[i] / this.wordCount;
		}
		for(int i = 0; i < Features.dutch.length; i++){
			this.input[index++] = (float)this.engCount[i] / this.wordCount;
		}
		sc.close();
		return this.input;
	} // getInput
} // class Neuron

