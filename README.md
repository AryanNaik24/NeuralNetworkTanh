# Neural Network from Scratch

This project implements a simple neural network from scratch using NumPy. The neural network is designed to learn the XOR function using two hidden layers and backpropagation.

## Features
- Implemented forward and backward propagation
- Uses **tanh** activation in hidden layers and **sigmoid** activation in the output layer
- He initialization for weight initialization
- Supports training with adjustable learning rate and epochs

## Prerequisites
Make sure you have Python installed along with NumPy. You can install NumPy using:
```sh
pip install numpy
```

## How to Use

1. Clone the repository:
   ```sh
   git clone https://github.com/AryanNaik24/NeuralNetworkTanh.git
   cd NeuralNetworkTanh
   ```

2. Run the script:
   ```sh
   python main.py
   ```

## Code Overview

### Activation Functions
- `tanh(x)`: Hyperbolic tangent function for non-linearity.
- `sigmoid(x)`: Sigmoid function used in the output layer.

### Initialization
- `initialize_weights_biases(layers)`: Initializes weights using He initialization and sets biases to zero.

### Forward Propagation
- `forward_propagation(X, weights)`: Computes activations for all layers using tanh for hidden layers and sigmoid for the output layer.

### Backward Propagation
- `backward_propagation(X, Y, weights, activation)`: Computes gradients using the chain rule and updates weights.

### Training the Model
- `train(X, Y, layers, learning_rate, epochs)`: Trains the neural network and prints loss every 500 epochs.

### Prediction
- `predict(X, weights)`: Computes the output predictions based on trained weights.

## Training on XOR Problem
The neural network is trained on the XOR dataset:

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
```

With the following architecture:
```python
layers = [2, 5, 3, 1]
```

## Expected Output
After training, the network should learn the XOR function and output:
```
Predictions: [0 1 1 0]
```

## Author
**Aryan Naik**  
GitHub: [@AryanNaik24](https://github.com/AryanNaik24)

## License
This project is open-source

