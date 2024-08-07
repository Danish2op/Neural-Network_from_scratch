# Neural Network Comparison for MNIST Dataset

## Overview
This project aims to compare the performance of neural networks using different activation functions (ReLU, LeakyReLU, ELU) for the MNIST dataset. The goal is to understand how these activation functions affect the learning and accuracy of the neural network model.

## Dataset
The MNIST dataset consists of grayscale images of handwritten digits (0-9), with each image being 28x28 pixels. The task is to classify these images into their respective digits.

## Implementation
### Data Preprocessing
- Loaded the dataset from 'train.csv'.
- Preprocessed the data by normalizing pixel values to the range [0, 1].

### Neural Network Architecture
- Input Layer: 784 neurons (28x28 pixels flattened)
- Hidden Layers: 128 neurons with ReLU/LeakyReLU/ELU activation function.
- Output Layer: 10 neurons with softmax activation for digit classification.

### Activation Functions
1. **ReLU (Rectified Linear Unit)**
   - $f(x) = max(0, x)$
   - Simple and effective, but can suffer from the "dying ReLU" problem.

2. **LeakyReLU**
   - $f(x) = max(\alpha x, x)$ where $\alpha$ is a small constant (e.g., 0.01).
   - Helps alleviate the "dying ReLU" problem by allowing small negative values.

3. **ELU (Exponential Linear Unit)**
   - f(x) = x if x > 0, and f(x) = α(e^x - 1) if x ≤ 0
   - Smooths the gradient near zero and can handle negative values well.

4. **Sigmoid**
   - $f(x) = \frac{1}{1 + e^{-x}}$
   - Outputs values between 0 and 1, often used for binary classification tasks.
   - Smooth and continuous, but can suffer from the vanishing gradient problem for very large or small inputs.

### Training
- Trained each model using gradient descent with a learning rate of 0.10 for 500 iterations.
- Calculated accuracy on a validation set every 10 iterations.
  
### Results
- **LeakyReLU + ELU**: Achieved accuracy of 89%.
- **ReLU**: Achieved accuracy of 88.45%.
- **Sigmod**: Achieved accuracy of 79.09%.

## Mathematics
### Softmax Activation

The softmax function is used in the output layer of neural networks for multiclass classification tasks. It converts raw scores (also known as logits) into probabilities, making it suitable for determining the likelihood of each class.

For a vector of raw scores $Z$, the softmax function calculates the probability of class $i$ (where $i$ ranges from 1 to $K$, the number of classes) as:

# σ(z)i = e^(zi) / Σ(e^(zi))

In this equation:
- $Z_i$ represents the raw score of class $i$.
- $e$ is the base of the natural logarithm.
- The denominator sums the exponentiated raw scores of all classes, ensuring that the resulting probabilities sum to 1.

The softmax function's output can be interpreted as the model's confidence in each class, with higher probabilities indicating higher confidence.


## Graphs
![Accuracy Graph](graphl.png)


## Conclusion
In conclusion, this project demonstrates the impact of different activation functions on neural network performance. Each activation function has its strengths and weaknesses, influencing the model's learning behavior and accuracy.


