# Multiple Logistic Regression

Here is my implementation of multiple logistic regression utilizing the sigmoid binding function, as well as batch gradient descent on the [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), predicting wether a given patient will have heart disease based on age, cholesterol and resting heart rate as model weights.

> By: [Oscar Sharaz Spencer](https://www.linkedin.com/in/oscar-sharaz/)

## Index
1. [Introduction](#introduction)
    - [Sigmoid Function](#sigmoid-function)
    - [Sigmoid Function Limits](#sigmoid-function-limits)
    - [Using the Sigmoid with Multiple Features](#using-the-sigmoid-with-multiple-features)
2. [Cost Function for Logistic Regression](#cost-function-for-logistic-regression)
    - [Binary Cost Function](#binary-cost-function)
    - [Gradient Descent](#gradient-descent)
3. [Making Predictions](#making-predictions)
4. [Binary Classification](#binary-classification)
5. [Running the Predictions](#running-the-predictions) 

---

## Introduction
When attempting to predict the values of a dataset containing binary results, linear regression is not a good choice. This is because a linear function cannot fit binary outputs (outputs of either 0 or 1) accurately. Linear regression assumes a continuous output, whereas binary classification problems require an output that is restricted to two possible values (0 or 1). Using a linear function in such cases can lead to inaccuracies, as it can predict values outside the desired range, such as negative values or values greater than one.

## Sigmoid Function
The sigmoid function is a more appropriate choice for binary classification tasks, as it maps input values to a range between 0 and 1. The sigmoid function is given by:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

## Sigmoid Function Limits
The limit evaluation of the sigmoid function as $x$ approaches infinity and as $x$ approaches negative infinity are indeed representing binary outputs:

$$
\lim_{{x \to \infty}} \sigma(x) = 1
$$

$$
\lim_{{x \to -\infty}} \sigma(x) = 0
$$

## Using the Sigmoid with Multiple Features
To predict binary outcomes with multiple input features, we can adapt the sigmoid function as follows (or **sigmoidize** is what I like to call it):

$$
\sigma(w_1x_1 + w_2x_2 + \cdots + w_nx_n + b) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + \cdots + w_nx_n + b)}}
$$

In matrix form, this can be represented as:

$$
\sigma(\mathbf{W}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{W}^T \mathbf{x} + b)}}
$$

## Cost Function for Logistic Regression
Logistic regression uses a specific cost function that is suitable for binary output values:

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

where:

$$
h_\theta(x) = \sigma(\theta^T x)
$$

## Binary Cost Function
The cost function can be split into two parts depending on the value of $y$, **I like to call this specific function a "toggle" function for the following reason**:

- When $y = 1$:  
  $$- \log(h_\theta(x))$$

- When $y = 0$:  
  $$- \log(1 - h_\theta(x))$$

## Gradient Descent
To optimize the cost function, we use gradient descent. The partial derivatives of the cost function with respect to $w$ and $b$ are:

For $w$:

$$
\frac{\partial J(\theta)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

For $b$:

$$
\frac{\partial J(\theta)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)
$$

These derivatives are then used to update $w$ and $b$:

$$
w_j := w_j - \alpha \frac{\partial J(\theta)}{\partial w_j}
$$

$$
b := b - \alpha \frac{\partial J(\theta)}{\partial b}
$$

where $$\alpha$$ is the learning rate.

## Making Predictions
After optimizing $w$ and $b$, predictions can be made for a new input $x$:

$$
\hat{y} = \sigma(w x + b) = \frac{1}{1 + e^{-(w x + b)}}
$$

For multiple input features:

$$
\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)}}
$$

Or, in matrix notation:

$$
\hat{y} = \sigma(\mathbf{W}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{W}^T \mathbf{x} + b)}}
$$

## Binary Classification
The resulting probability $\hat{y}$ can be converted into a binary prediction using a threshold (commonly 0.5):

$$
\text{Prediction} =
\begin{cases}
1 & \text{if } \hat{y} \geq 0.5 \\
0 & \text{if } \hat{y} < 0.5
\end{cases}
$$

This approach enables the model to output binary predictions after fitting the sigmoid function with the optimal weights and bias obtained through gradient descent.

## Running the Predictions

### main.py
The `main.py` file is the main entry point for running the logistic regression predictions. It initializes the dataset and runs the gradient descent algorithm on the given cost function, and outputs the predicted parameter error values into a visualizer.

### Installation and Execution
To run the predictions, follow these steps:

1. Ensure you have Python 3 installed on your system.
2. Clone this repository and navigate to the project directory.
3. Run the following commands to install any necessary dependencies and execute the script:

```bash
# Clone the repository
git clone https://github.com/oskccy/logistic-regression-from-scratch.git
cd logistic-regression-from-scratch

# Install dependencies
pip install -r requirements.txt

# Run the predictions
python3 main.py
