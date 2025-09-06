'''
    Notes for the IBM developer article "Neural networks from Scratch"
    https://developer.ibm.com/articles/neural-networks-from-scratch/
'''

from sklearn.datasets import fetch_openml
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
# import cupy as np # Runs extremely slower than numpy

import time

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=3, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer),
            'change_w': {},
            'error_w': {}
        }

        return params

    def forward_pass(self, x_train):
        # print(f"Starting forward pass...")
        params = self.params

        '''
            W1: 128x784
            W2: 64x128
            W3: 10x64

            ZI/A1: 128x1
            Z2/A3: 64x1
            Z3/A3: 10x1

            Hidden layers use sigmoid for activation, whereas the output layer uses softmax.
        '''

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        # print(f"Ending forward pass...")

        return params['A3']

    def backward_pass(self, y_train, output):
        # print(f"Starting backward pass...")

        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params

        # Experiment with caching error and change_w

        # Calculate W3 update
        '''
            - Deriatives are enabled during backpropogation
            - output: 10x1
            - y_train: 10x1
            - error (outgoing): 10x1
            - params['A2']: 64x1
            - change_w['W3']: np.outer(error, params['A2']): np.outer(10x1, 64x1) -> 10x64
        '''
        params['error_w']['W3'] = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        params['change_w']['W3'] = np.outer(params['error_w']['W3'], params['A2'])

        # Calculate W2 update
        '''
            - Deriatives are enabled during backpropogation
            - error (incoming): 10x1
            - params['W3'].T: 64x10
            - params['W3'].T x error: 64x1
            - error (outgoing): 64x1
            - params['A1']: 128x1
            - change_w['W2']: np.outer(error, params['A1']): np.outer(64x1, 128x1) -> 64x128
        '''
        params['error_w']['W2'] = np.dot(params['W3'].T, params['error_w']['W3']) * self.sigmoid(params['Z2'], derivative=True)
        params['change_w']['W2'] = np.outer(params['error_w']['W2'], params['A1'])

        # Calculate W1 update
        '''
            - Deriatives are enabled during backpropogation
            - error (incoming): 64x1
            - params['W2'].T: 128x64
            - params['W3'].T x error: 128x1
            - error (outgoing): 128x1
            - params['A0']: 784x1
            - change_w['W1']: np.outer(error, params['A0']): np.outer(128x1, 784x1) -> 128x784
        '''
        params['error_w']['W1'] = np.dot(params['W2'].T, params['error_w']['W2']) * self.sigmoid(params['Z1'], derivative=True)
        params['change_w']['W1'] = np.outer(params['error_w']['W1'], params['A0'])

        # print(f"Ending backward pass...")

        # return change_w

    def update_network_parameters(self):
        # print(f"Starting update of network params...")
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in self.params['change_w'].items():
            self.params[key] -= self.l_rate * value # The gradient of the descent function is 1?

        # print(f"Ending update of network params...")

    def compute_accuracy(self, x_val, y_val):
        # print(f"Starting computation of accuracy...")

        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        # print(f"Ending computation of accuracy...")

        return np.mean(np.array(predictions))

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs): # No of times we run through the entire dataset
            for x,y in zip(x_train, y_train): # A pair of inputs and the corresponding output
                output = self.forward_pass(x)
                # changes_to_w = self.backward_pass(y, output)
                self.backward_pass(y, output)
                self.update_network_parameters()

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))

def main():
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    x = (x/255).astype('float32')
    y = to_categorical(y)

    # print(f"type(x): {type(x)}, type(y): {type(y)}")

    '''
        Splits the dataset into training ("_train") and validation ("_val") sets.
        x_train and y_train go together
        X_val and y_val go together
    '''
    x_train, x_val, y_train, y_val = train_test_split(x.values.astype("float32"), y, test_size=0.15, random_state=42)

    # print(f"type(x_train): {type(x_train)}, type(y_train): {type(y_train)}, type(x_val): {type(x_val)}, type(y_val): {type(y_val)}")

    # x_train_cupy = np.array(x_train)
    # x_val_cupy = np.array(x_val)
    # y_train_cupy = np.array(y_train)
    # y_val_cupy = np.array(y_val)

    # print(f"len(x_train_cupy): {len(x_train_cupy)}, len(y_train_cupy): {len(y_train_cupy)} \
    #        len(x_val_cupy): {len(x_val_cupy)}, len(y_val_cupy): {len(y_val_cupy)}");

    # print(f"len(x_val[0]): {len(x_val[0])}")
    # print(f"len(y_val[0]): {len(y_val[0])}")

    dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
    dnn.train(x_train, y_train, x_val, y_val)

if __name__ == "__main__":
    main()