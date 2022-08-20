import numpy as np
# perceptron is a NN with no hidden layers
# following a Youtube tutorial from Polycode
# https://www.youtube.com/watch?v=kft1AJ9WVDk
# this will be our normalizing function phi(x)
def sigmoid(x):
    """ normalizing funciton that returns number between 0 and 1 """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array(
    [
        [0,0,1],
        [1,1,1],
        [1,0,1],
        [0,1,1]
    ]
)

training_outputs = np.array([[0,1,0,1]]).T
# seems incapable of learning xor as mentioned in the perceptron wikipedia entry
xor_inputs = np.array(
    [
        [0,0,1],
        [1,1,1],
        [1,0,1],
        [0,1,1]
    ]
)

xor_outputs = np.array([[0,0,1,1]]).T
training_inputs = xor_inputs
training_outputs = xor_outputs

# not really 100 sure why we need these yet
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3,1)) - 1 
print(f'random staring weights\n{synaptic_weights}')

for itter in range(20000):

    input_layer = training_inputs

    # forward prop
    # sigmoid(x1*w1 + x2*w2 + ... + xn*wn))) where n is the number of input nodes 
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # out put y-hat (our current best aproximation of the true value y)
    error = training_outputs - outputs

    #calculate how much we need to adjust the weights
    adjustments = error * sigmoid_derivative(outputs)
    # we need -- dot | for linear algebra to work, so we take a transpose
    adjust_weight_by = np.dot(input_layer.T, adjustments)

    synaptic_weights += adjust_weight_by

print(f"synaptic weights after trainging\n {synaptic_weights}")

# want the outputs to look like training output?
print(f"output after training:\n{outputs}")

# general training process

# take inputs from training example and 
# run them through our current formula to get the output
# (forward propigation?)

# calculate the error

# adjust weights depending on the severity of the error

# repeat 20k times

# ERROR WEIGHTED DERIVATE
# adjust weight by = error * input * phi'(output) 
# error = output - actual output
# input = 1 or 0

# if the output was a large negative or positive then that indicates high confidence
# if we look at the derivative of phi, phi', 
# then we see large inputs give small gradients, 
# thus we dont adjust those weights as much
# conversley??
# if the output was small, that means low confidence, 
# and the gradient of phi at that point is larger, 
# indicating that we need to make a larger adjustment
