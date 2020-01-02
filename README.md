# Diabetes Neural Network

#### By:Mokshda Sharma
Data used to train this model: https://datahub.io/machine-learning/diabetes

#### Architecture:
Input Neurons:8<br/>
(Input parameters:NumberOfPregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)<br/>
Output Neurons:1(Diabetic/Non-diabetic)<br/>
Hidden Layers:1<br/>
Activation Function:Sigmoid<br/>

This Neural Network is feedforward and fully connected.Every node in a given layer is fully connected to every node in next layer.


## Flow of Program

#### Files:
There are two files,one containing the neural network and other containing the dataset to train the network.The file Diabetes_diagnosis_nn.py contains the neural network.
 
 
#### Introduction:
The Neural Network is programmed in Python mainly using the NumPy library.NumPy(Numerical Python) consists of  Mutidimensional array objects and a collection of routines for processing those arrays.<br/>

diabetes_diagnosis_nn.py code breakdown:

The following lines import all libraries and dependencies used:
```ruby 
from numpy import exp, array, random, dot
import numpy as np
```

Then data from csv file is converted into arrays:
```ruby
import csv

preg = [] #number of pregnancies
glu = []  #Glucose
Blood = [] #blood pressure
Skin_Thickness = [] #skin thickness
Insulin = []
BMI = []
DPF = [] #diabetes pedigree function
Age = []



with open(r'C:\Users\hp\Downloads\diabetes.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        preg.append(int(row[0]))
        glu.append(int(row[1]))
        Blood.append(int(row[2]))
        Skin_Thickness.append(int(row[3]))
        Insulin.append(int(row[4]))
        BMI.append(float(row[5]))
        DPF.append(float(row[6]))
        Age.append(int(row[7]))
 ```
 
 Create normalization and denormalization functions for input parameters:
 ```ruby
 def nor_denor_preg(v):
    v=int(v)
    if(v>1):
        normal = (v-min(preg))/(max(preg)-min(preg))
        return normal
    else:
        denormal = v*(max(preg)-min(preg)) + min(preg)
        return denormal
   ```
The normalization and denormalization functions for rest of the parameters are created in a similar manner<br/>
   
Then initialize input and output arrays with random data and append the data from csv file as elements of the arrays inside NumPy array.
```ruby
with open(r'C:\Users\hp\Downloads\diabetes.csv') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
npa = np.asarray(results, dtype=np.float32)    
print(npa)         
count = 0  
temp_arr=[0.5]
with open(r'C:\Users\hp\Downloads\outputNew.csv.xlsx') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp = int(row[0])
        temp_arr[0] = temp
        out_arr[count] = temp_arr
        count = count + 1
```   
Define function to assign Weights to the layers inside neural network.This fuction randomly assigns weights.
```ruby
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
```
Define the layers of neural network i.e. hidden layer(Layer 1) and output layer(Layer 2)
```ruby
class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
 ```
Sigmoid Function is used as activation function to train the network.It is a S shaped curve whose value lies between 0 and 1.Pass the weighted sum of the inputs through this function to normalize them between 0 ad 1.
 ```ruby
 def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
 ```
 Define the derivative of sigmoid function.This function is differentiable everywhere on the curve.
 ```ruby
 def __sigmoid_derivative(self, x):
        return x*(1 - x)
 ```
Train the neural network by backpropagation in which the weights are repeatedly adjusted to minimize the difference between actual output and desired output.
 ```ruby
 def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
 ```
 Pass the training set through neural network:
 ```ruby
 output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)
 ```
Calculate the error for output layer by taking the difference between actual and desired output and define the derivative of the sigmoid function
 ```ruby
 layer2_error = training_set_outputs - output_from_layer_2
 layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
 ```
Calculate the error for layer 1.By analyzing the value of weights in layer 1,the contribution of layer 1 to the error i layer 2 is found
 ```ruby
  layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
  layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)
 ```
 To calculate the adjustment required in the values of weights:
 ```ruby
 layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
 layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
 ```
  Adujst the weights:
  ```ruby
  self.layer1.synaptic_weights += layer1_adjustment
  self.layer2.synaptic_weights += layer2_adjustment
  ```
  Define a function to take the dot product of synaptic weights with the inputs of respective layers:
  ```ruby
  def think(self, inputs):
       output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
       output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
 ```
 Print the weights:
 ```ruby
 def print_weights(self):
        print( "    Layer 1(Hidden) : ")
        print(self.layer1.synaptic_weights)
        print( "    Layer 2 :")
        print(self.layer2.synaptic_weights)
 ```
 
 Create Layer 1 having 8 inputs and 6 neurons and Layer 2 having a single neuron with 6 inputs:
 ```ruby
 layer1 = NeuronLayer(6, 8)
 layer2 = NeuronLayer(1,6)
 ```
 Combine the layers to create a neural network:
 ```ruby
  neural_network = NeuralNetwork(layer1, layer2)
  ```
  Train the neural network using the training set,doing it 6,00,000 times and making small adjustments each time:
 ```ruby
  neural_network.train(training_set_inputs, training_set_outputs, 60000)
 ```
 Print new synaptic weights after training:
 ```ruby
 neural_network.print_weights()
 ```
 Test the neural network with a new situation:
 ```ruby
  print( "Stage 3) Considering a new situation --- ")
    
    test = []
    with open(r'C:\Users\hp\Downloads\diabetes.csv') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            test.append(row)
    input_test = np.asarray(test, dtype=np.float32)
    out_plot_arr=np.array([[.5],[.5],[.2],[.5],[.5],[.2],[.5],[.5],[.2],[.5],[.9]])
```
 Print the output
 ```ruby
  hidden_state, output = neural_network.think(array(input_test))
  print(output)
 ``` 
This is the neural network to predict Diabetes.To increase the accuracy of the network we can either increase the number of hidden layers or the input neurons in each layer.Increasing the size of training data also improves the accuracy of the model.        
   
        
        





 
