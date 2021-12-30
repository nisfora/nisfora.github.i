---
title: "Deep Neural Network from Scratch - Part 1"
date: 2019-04-18T15:34:30-04:00
categories:
  - blog
tags:
  - Jekyll
  - update
classes: wide
header:
  teaser: https://unsplash.com/photos/842ofHC6MaI
# layout: splash
# classes:
#   - landing
#   - dark-theme
---

In this post I will be explaining how to design your own Neural Network (Keras-like) framework from scratch in python 3.7+. This post assumes you have basic understanding of how a Neural Network works and familiarity with forward-backward propagation, Loss functions, Optimizers, Regularization and so on. Without any further due, let’s jump right into it.

![Photo by [Moritz Kindler](https://cdn.hashnode.com/res/hashnode/image/upload/v1620220626760/1w0avz-fZ.html) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/12000/0*AMlugSvbFgAH4p3t)*Photo by [Moritz Kindler](https://unsplash.com/@moritz_photography?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)*

A basic neural network model implementation in **Keras** framework looks like the following. In our implementation from scratch, we will try to design a similar framework as Keras (but basic).

```python
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)

```

## **Base Neural Network class**

In order to keep all the functional elements together we will create a class (Vanilla here) to organize the functionality of our Neural Network. The class can store variables to define how the network in structured and parameterized.

```python
class Vanilla:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.alpha_0 = 0.01
        self.loss_fn = "entropy"
        self.iterations = 0
        self.dict = {}
        self.decay_rate = 0.001
        self.problem_type = "c" # c for classification & r for regression
```

## **Adding Layers**

This function will allow users to add layers and design the graph of the network. Note that this function stores the number of hidden units in a variable named layers (similarly activation).

```python
def add_layer(self, units, input_dim=0, activation="sigmoid"):
  if input_dim == 0:
      self.layers.append(units)
      self.activations.append(activation)
  else:
      self.layers.append(input_dim)
      self.layers.append(units)
      self.activations.append(activation)
```

## **Compile Function**

This function initializes various model parameters before starting the training in the next step.

```python
def compile(self, learning_rate=0.01, decay_rate=0.001, loss="entropy"):
        self.alpha_0 = learning_rate
        self.loss_fn = loss
        self.decay_rate = decay_rate
```

## **Training the network**

The fit method lies at the heart of this implementation. Although the actual function has a vast functionality, we will look at the most important part which is forward and backpropagation.

```python
# Forward Pass
for i in range(no_of_layers):
    a.append(self.forward(
        a[i], self.dict['W' + str(i+1)], self.dict['b' + str(i+1)], activation=self.activations[i]))

dz = []
dW = []
db = []
dz.append(y_train)

# Backpropagation of Gradients
for i in range(no_of_layers, 0, -1):
    if i == no_of_layers:
        [dzz, dWW, dbb] = self.backward(
            a[i-1], a[i], [], dz[no_of_layers-i], activation=self.activations[i-1], output_layer=True)
    else:
        [dzz, dWW, dbb] = self.backward(
            a[i-1], a[i], self.dict['W' + str(i+1)], dz[no_of_layers-i], activation=self.activations[i-1])

    dz.append(dzz)
    dW.append(dWW)
    db.append(dbb)

# Update Weights and Biases
for i in range(1, no_of_layers + 1):
    if regularize:
        dW[no_of_layers -
            i] += self.l1_reg(self.dict['W' + str(i)], lambda_)

    self.dict['W' + str(i)] -= (alpha * dW[no_of_layers-i])
    self.dict['b' + str(i)] -= (alpha * db[no_of_layers-i])
```

## **Activation Functions**

There are several activation functions (relu, sigmoid, softmax, tanh) defined in original Vanilla. An example of sigmoid and it’s derivative is shown below.

```python
def sigmoid(self, x):
    return (1.0/(1.0+np.exp(-x)))

def d_sigmoid(self, x):
    a = self.sigmoid(x)
    return a * (1 - a)
```

## **Loss Functions**

Vanilla class contains three different loss functions — Cross Entropy, Logistic Loss, Mean Squared Error (MSE). An example implementation of Cross entropy is shown below.

```python
def cross_entropy_loss(self, y_train, y_hat, epsilon=1e-11):
      m = y_train.shape[0]
      n = y_train.shape[1]

      y_hat_clip = np.clip(y_hat, epsilon, 1 - epsilon)
      result = ((-1.0 / (m)) * np.sum(np.sum(y_train *
                np.log(y_hat_clip), axis=1), axis=0))

      if (np.any(np.isnan(result))):
          print("Error in Cross Entropy")
          exit()
      return result
```

## **Predicting the output**

The predict method takes the input data (X) and performs a forward pass through the network in order to get prediction values (y).

```python
def predict(self, X):
      no_of_layers = len(self.layers) - 1
      a = []
      a.append(X)

      for i in range(no_of_layers):
          a.append(self.forward(a[i], self.dict['W' + str(i+1)],
                   self.dict['b' + str(i+1)], activation=self.activations[i]))
      return a[-1]
```

This is a developing post and more details will be added soon. If you found this post helpful, follow me and share this post!

Github Link: [Vanilla Neural Network Repository](https://github.com/nitesh4146/Vanilla-Neural-Network)

Cheers.