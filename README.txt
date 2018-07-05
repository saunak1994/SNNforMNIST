Requires "mnist_train.csv" and "mnist_test.csv" . These are the original MNIST (http://yann.lecun.com/exdb/mnist/) 
datasets in a csv format. Available at:

https://drive.google.com/open?id=1b_uP0fMelgH_RD9ODNoLw5OR44l9ovya
https://drive.google.com/open?id=1gbLr38NHArKWIRG27O6tAo_-y9VLI98Y


#-----------------------------------------------------------------
# Parameter Generator
#-----------------------------------------------------------------

To run code: 


$ python paramGenerator.py <parameter name> <start_value> <end_value> <step_value>


This will generate parameter files having default parameter values and concerned parameter ranging from start value to 
end value (both included) each separated by the step value and output the number of simulations required to loop over all
the files. 



#-----------------------------------------------------------------
# Spiking Neural Network Trainer
#-----------------------------------------------------------------

To run code: 


$python SNNTrainRoutine.py <start_value> <end_value> <start_mode> <operation_mode> <parameter name> <number of simulations>


This will train the file over the training examples starting from <start_value> till (excluding) the <end_value>.

<start_mode> is either: 

1. "start": In which case, the training will start from scratch (initial weights). 

2. "continue": In which case, the training script will look for a file "trainedWeights.csv" containing already trained weights
upto some point and continue training from there. Allows training in laps or checkpoints. 

 

<operation_mode> is either: 

1. "dse" (for Design space exploration)
In which case, you have to provide the parameter name that is
to be investigated and also the number of simulations, as suggested by the paramGenerator.py script. The resulting 
receptive fields will be saved in a monolithic pdf file, and trained weights will be saved in separate files corresponding
to each parameter file previously generated. 

II. "standalone" (for standalone training)
In which case, you should ignore the subsequent arguments.
The resulting single receptive field figure will be saved in a pdf file and a single trainedWeights.csv file will be
generated with the trained Weights in it.  

For example: to run 10 simulations on different values of the "tau_theta" parameter:

$python SNNTrainRoutine.py 0 20000 start dse tau_theta 10 

to run standalone training from an already trained checkpoint:

$python SNNTrainRoutine.py 20000 30000 continue standalone










[NB: Project still in development. Contact saha@iastate.edu for further info]

07.04.2018
© Author: Saunak Saha
Iowa State University