import numpy as np

def main():
    mat1 = np.array([[2, 3, 4], 
                     [5, 6, 7],
                     [8, 9, 10]], np.float32)
    np.identity(3, np.float32)
    print(f"mat1: {mat1}")

if __name__ ==  "__main__":
    main()


def forward_pass(self, x_train):
    params = self.params

    # input layer activations becomes sample
    params['A0'] = x_train # 784x1

    # input layer to hidden layer 1
    params['Z1'] = np.dot(params["W1"], params['A0']) # 128x1
    params['A1'] = self.sigmoid(params['Z1']) # 128x1

    # hidden layer 1 to hidden layer 2
    params['Z2'] = np.dot(params["W2"], params['A1']) # 64x1
    params['A2'] = self.sigmoid(params['Z2']) # 64x1

    # hidden layer 2 to output layer
    params['Z3'] = np.dot(params["W3"], params['A2']) # 10x1
    params['A3'] = self.softmax(params['Z3']) # 10x1

    return params['A3']