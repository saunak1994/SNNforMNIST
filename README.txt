#-----------------------------------------------------------------
# Parameter Generator
#-----------------------------------------------------------------

To run code: 


$ python paramGenerator.py <parameter name> <start_value> <end_value> <step_value>


This will generate parameter files having default parameter values and concerned parameters ranging from start value to 
end value (both included) each seperated by the step value and output the number of simulations required to loop over all
the files. 



#-----------------------------------------------------------------
# Spiking Neural Network Trainer
#-----------------------------------------------------------------

To run code: 


$python SNNTrainRoutine.py <start_value> <end_value> <simulation mode> <parameter name> <number of simulations>


This will train the file over the training examples starting from <start_value> till (excluding) the <end_value>. If 
training begins from example 0 (the first example of training data), the network will be built from scratch. Otherwise,
the script will restore an older network state from an 'SNNetState' file in the current directory. 

Modes of operation: 

I. Simulation mode (for Design space exploration)
In this mode, the <simulation mode> value should be "True". In that case, you have to provide the parameter name that is
to be investigated and also the number of simulations, as suggested by the paramGenerator.py script. The resulting 
receptive fields will be saved in a monolithic pdf file, and trained weights will be saved in seperate files corresponding
to each parameter file previously generated. 

II. Standalone mode 
In thus mode, the <simulation mode> value should be "False". In that case, you should ignore the subsequent arguments.
The resulting single receptive field figure will be saved in a pdf file and a single trainedWeights.csv file will be
generated with the trained Weights in it. This will be required by the NeuronClassAssignment.py script. 

For example: to run 10 simulations on different values of the "tau_theta" parameter, run: 

$python SNNTrainRoutine.py 0 20000 True tau_theta 10 


#-----------------------------------------------------------------
# Neuron Class Assignment
#-----------------------------------------------------------------




