import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()

        # A NN with 5 layers. This includes the activation units.
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_features, 30), # by default, `bias is true
            torch.nn.ReLU(), # activation unit
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, output_features) 
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

if __name__ == '__main__':
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)

    print(f'Neural network: {model}, layers: {len(model.layers)}')
    for i,layer in enumerate(model.layers):
        if type(layer) != type(torch.nn.ReLU()):
            print(f'Layer {i}, weight: {layer.weight.shape}, bias: {layer.bias.shape}')

    '''
    We compute the number of trainable params, which is essentially the number
    of elements in the weight and bias tensors of each of the layers. The ReLU
    layer is excluded from trainable params.
    '''
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'Total number of trainable params: {trainable_params}')

    '''
    The linear transformation is applied as `y = xA^T + b`, where A is the
    weight and A^T is transpose. This is opposed to `y = Ax + b`.
    Hence, `X` is a 1x50 matrix, as opposed to 50x1.
    '''
    X = torch.rand(1, 50) 

    # No need to store gradients, as we are not doing back-propogation yet.
    with torch.no_grad():
        Y = torch.softmax(model.forward(X), dim=1)
    
    print(f'Y: {Y}')
