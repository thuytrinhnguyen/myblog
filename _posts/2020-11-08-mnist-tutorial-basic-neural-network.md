---
layout: post
title: "MNIST Tutorial - Neural Networks approach from scratch in Python"
date: 2020-11-08 14:08
category: deeplearning
tags: neural-networks deeplearning mnist
---

> A neural network = Forward path + Backward path. Simple, isn't it? Let's build one from scratch and classify handwritten digits.

<!--more-->

These days, popular programming frameworks offer users convenient functions to construct a variety of neural networks using a few lines of code. But sometimes knowing how things work behind the scene can be interesting so in this tutorial, I will show you how to build a minimal neural network from scratch in Python. The objective is to use this model to classify handwritten digits from the common dataset MNIST.

{% figure caption: "Fig. 1. Sample images from MNIST dataset (Image source: [Wikiwand](https://www.wikiwand.com/en/MNIST_database))." %}
![MNIST Sample Images]({{'/assets/images/mnist-dataset.png'}})
{% endfigure %}

<mark><b>Highlights</b></mark> 

{: class="table-of-content"}
* TOC
{:toc}

## Overview 

We will construct a minimal Neural network with one hidden layer and a 10-class softmax classifier output layer. My advice is to draft out the formulas and dimensions of all parameters beforehand which will help us debug the codes easier. The plan is to __(1)__ load the dataset, __(2)__ build the forward path, __(3)__ code the backward path and __(4)__ combine everything into a model. Along the way, we will define some helper functions.

{% figure caption:"Fig. 2. The minimal Neural Network has one hidden layer and a softmax classifier." %}
![Neural Network Architecture]({{'/assets/images/mnist-tutorial-nn-ach.png'}})
{% endfigure %}

## Step-by-step Implementation

### Step 1: Download dataset
You can download __MNIST dataset__ from various sources, I download mine from [Kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv). Simply save and unzip the package. There are two files in the folder: _mnist_train.csv_ and _mnist_test.csv_, each contain both the __features (X)__ and the __label (Y)__. 

### Step 2: Load the data
Firstly, we need to separate __features (X)__ and __label (Y)__ and perform some data preprocessing operations on them. The label is the first column followed by 784 columns of image pixels (i.e. X are 28x28 images) in both the training and testing sets. The training set has 60,000 entries while the testing set has 10,000. 

Because X are grayscale images, the pixel values vary greatly from 0 to 255, we can normalize them by dividing all pixels by 255. On the other hand, we need to convert the labels into one-hot vectors to classify the 10 digits (0 - 9). I will summarize the process in Fig. 3, the same operations are applied to the test set. 
 
{% figure caption:"Fig. 3. demonstrates data preparation and dimensions of X and Y in MNIST training dataset." %}
![Data]({{'/assets/images/mnist-tutorial-nn-data.png'}})
{% endfigure %}

Here is a handy code snippet to convert the labels into one-hot embeddings:

```python
Y_train = np.zeros((10, X_train.shape[1]))
Y_train[Y_train_idx, np.arange(Y_train_idx.size)] = 1
```

### Step 3: Helper functions

__Initialize parameters__

All the weights (W) should have dimensions of $$(n^{[l]}, n^{[l-1]})$$, where $$n^{[l]}$$ is the number of neurons in the current layer and $$n^{[l]}$$ is that of the previous one. I recommend applying [He initialization](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78) to optimize the initializer. For the biases, we can initialize them to be zero vectors with the dimension of the current layer.

Note: The way I set <code>layers = [50, 10]</code> does not include the dimension of the Input layer (X), therefore, I specify two cases in the code.

```python
def init_params(X, layers):
    W, b = {}, {}
    for l in range(len(layers)):
        if l == 0:
            W[l] = np.random.randn(layers[l], X.shape[0]) * np.sqrt(2/X.shape[0]) 
        else:
            W[l] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2/layers[l - 1])
        b[l] = np.zeros((layers[l], 1))
    return W, b
```

__ReLU__

The ReLU function simply outputs Z if Z is positive and 0 otherwise. You can make use of the <code>maximum()</code> function in numpy library. 

__Softmax__

We will use Softmax activation to generate the probability for each digit in the 10-dimension vector. We can construct the Softmax activation using the formula:

$$
A^{L}_{i} =  \frac{e^{z^{L}_{i}}}{\sum_{j=1}^{N}e^{z^{L}_{j}}}
$$

where $$A^{L}_{i}$$ is the probability of each node and K is the number of nodes/classifiers of the last layer.

__Compute cost__

The cost function of Softmax for one example (for K classes) is:

<mark>$$ - \sum_{j=1}^{K} y_j \; log (\hat{y}_j)$$</mark> 

To calculate the loss, sum the cost of all training examples. This function serves two purposes. Firstly, by printing the cost, we can observe the trend of the model. If the cost does not decrease gradually, this means there are probably some bugs in the code. Secondly, because the cost function fluctuates across iterations, by tracking the training and testing (validating) cost, we can decide on the optimal number of epoch that minimize the cost of testing dataset. 

__Update parameters__

To keep things simple, in the tutorial, I will apply Gradient Descent to optimize the weights and biases for each layer:

```python
def update_params(W, b, dW, db, L, lr):
    for l in range(L):
        W[l] = W[l] - lr * dW[l]
        b[l] = b[l] - lr * db[l]
    return W, b
```

__Predict__

Evaluating the accuracy of the model can be as straightforward as run a forward round using the _optimal_ weights and biases and compare the results to the ground truth labels. Note that since the outputs of the model are 10-dimensional vectors, we need to transform the outcomes to its single-digit form as follows:

```python
def predict(X, Y, W, b, L):
    _, A = forward(X, W, b, L)
    Y_pred = np.argmax(A[L - 1], axis=0) 
    accuracy = np.sum(Y_pred == Y) / Y.shape[1]
    return accuracy
```

### Step 4: Forward path

To calculate the forward path, we calculate Z and A of layer $$[l]$$ using the formula:

$$Z^{[l]} = W^{[l]} \times A^{[l-1]} + b^{[l]} $$

$$A^{[l]} = g^{[l]}(Z^{[l]}) $$

where $$g^{[l]}$$ is the activation of layer $$l$$ (ReLU and Softmax).

We store Z and A as dictionaries to access them later for the backward path and to make prediction. Remember that our model only has one 50-unit hidden layer and a 10-class softmax layer, we define <code>layers = [50, 10]</code>. Therefore, we set the first previous <code>A_prev = X</code>. For the iteration, we will pass in <code>L = len(layers)</code>.

```python
def forward(X, W, b, L):
    Z, A = {}, {}
    A_prev = X                                  # Initiate X as A[0]
    for l in range(L):
        if l > 0:
            A_prev = A[l - 1]                   # Update A of previous layers
        Z[l] = np.dot(W[l], A_prev) + b[l]
        if l == L - 1:            
            A[l] = softmax(Z[l])
        else:
            A[l] = relu(Z[l])
    return Z, A
```

### Step 5: Backward path

To update the parameters (W and b), we need to find the derivative of the loss function with respect to each of the parameter. This can be done using the following formulas:

$$
dZ^{[L]} = A^{[L]} - Y
$$

$$
dZ^{[l]} = dA^{[l]} \times g^{[l]'}(Z^{[l]})
$$

$$
dW^{[l]} = \frac{1}{m} dZ^{[l]} \times A^{[l-1]}.T
$$

$$
db^{[l]} = \frac{1}{m} sum(dZ^{[l]})
$$

$$
dA^{[l]} = W^{[l+1]}.T \times dZ^{[l+1]}  
$$

Because <code>layers = [50,10]</code>, <code>range(L)</code> means to run from 0 to L-1 (i.e. layer 0 (hidden layer) and 1 (output layer)). The reversed sequence will access data from layer L-1 to 0 as follows: 

```python
def backward(X, Y, Z, A, W, L, m):
    dA, dZ, dW, db = {}, {}, {}, {}
    for l in reversed(range(L)):
        if l == 0:
            A_prev = X
        else:
            A_prev = A[l - 1]
        if l == L - 1:  # Calculate dZ for the last layer separately from others (as it uses Softmax activation)
            dZ[l] = A[l] - Y
        else:
            dA[l] = np.dot(W[l + 1].T, dZ[l + 1])
            dZ[l] = dA[l] * np.where(Z[l] < 0, 0, 1)  # Derivative of Relu activation function = 0 if Z < 0 and = 1 otherwise
        dW[l] = 1 / m * np.dot(dZ[l], A_prev.T)
        db[l] = 1 / m * np.sum(dZ[l], axis=1, keepdims=True)
    return dW, db
```

### Step 6: Putting it all together 

Now we have had all the components required to build the model. The objective is to train the model with training data and use the optimal weights and biases to make prediction for the testing data. To capture the optimal parameters, we choose those that minimize the test set cost. Later, we use these optimal values and run <code>predict</code> function to calculate Test Accuracy.

__Side Note:__ 

* When training the model for hundreds or thousands of iterations, it is handy to have some visualization assistance which shows a progress bar and estimates the remaining time to complete training. In this post, I imported <blue><code>tqdm</code></blue> library for that purpose.

* You can easily experiment with other fully-connected neural network architectures by changing <code>layers = [ ]</code>.


```python
def train():
    X_train, Y_train, X_test, Y_test, Y_train_idx, Y_test_idx = load_data()

    layers = [50, 10]
    epoch = 2000
    lr = 0.05

    L = len(layers)
    W, b = init_params(X_train, layers)
    m = X_train.shape[1]
    n = X_test.shape[1]
    pbar = tqdm(range(epoch))
    min_epoch = 0
    min_cost = 100
    W_opt = 0
    b_opt = 0

    for i in pbar:                            
        Z, A = forward(X_train, W, b, L)
        dW, db = backward(X_train, Y_train, Z, A, W, L, m)
        W, b = update_params(W, b, dW, db, L, lr)
        cost_train = compute_cost(A[L - 1], Y_train, m)

        if i%5 == 0:
            Z_test, A_test = forward(X_test, W, b, L)
            cost_test = compute_cost(A_test[L - 1], Y_test, n)
            if cost_test < min_cost:
                min_cost = cost_test
                min_epoch = i
                W_opt = copy(W)
                b_opt = copy(b)
        pbar.set_description(f"Cost Train = {round(cost_train, 4)}  -  Cost Test = {round(cost_test, 4)}")

    test_acc = predict(X_test, Y_test_idx, W_opt, b_opt, L)
    print(f'Optimal Epoch: {min_epoch}  -  Test Accuracy = {round(test_acc, 4)}')
```

## Wrap-up Highlights

Voila! Now you know how to construct a minimal Neural Network to predict handwritten digits. Although our model is simplistic, the results are pretty decent with __94% accuracy__ on the test data. This post is a stepping stone for you to customize your model. In reality, people rarely have to code from scratch thanks to the convenience programming frameworks offer. Hopefully, this sneakpeek into how basic functions work behind the scene can be useful to you in some way.

Full source code is available on my [Github]().


## References

{% bibliography --cited %}