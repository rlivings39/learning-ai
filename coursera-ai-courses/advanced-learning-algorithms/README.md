# Advanced Learning Algorithms

Course link: https://www.coursera.org/learn/advanced-learning-algorithms/

Neural networks began as an attempt to mimic the way that the brain works. They mimic the brain in being an input/output system that has many interconnections with pieces that process.

Modern neural networks made great strides in speech recognition, then images and computer vision, and finally text (NLP)

Neural networks have taken off recently for a few reasons. The amount of digital data we have is massive. Traditional learning algorithms like linear and logistic regression. The performance of NNs is better and sometimes scales up by increasing the size of the network.

## How neural networks operate: demand prediction example

Consider you're planning to sell a shirt and you want to predict if it will be a top seller.

Suppose you have a system that takes the price and outputs if the shirt will be a top seller. If you use the sigmoid algorithm you can view a neuron as a computer that takes in the price and spits out the output probability (a.k.a. activation)

Consider the same problem with features (price, shipping cost, marketing, material). We may think that variables are (affordability, awareness, perceived quality). We make a neuron for each of those 3 variables. Each of those uses one or more input features. Their outputs are wired to a final neuron.

The three neurons are a **layer** which is a grouping of neurons taking the same or similar inputs and outputting values. The final layer is called the **output** layer. The output values of the affordability, awareness, perceived quality are called the **activations** or **activation values**.

The first layer of input numbers is called the **input layer**

In this example we manually wired up the inputs and neurons. In practice every neuron will have access to every input from the prior layer. Training will then downplay the useless inputs rather than a human needing to do so manually.

The input layer will be written as a vector that's fed to the next layer. The middle layers output activation vectors which then become inputs to the subsequent layer.

Layers in the middle are often called **hidden layers** because the correct values for those are not present in the training set.

**Important** In our example, the hidden layer is just 3D logistic regression mapping (affordability, awareness, perceived quality) -> probability of being a top seller. However, the learning algorithm helps us learn what features are important and useful.

Neural networks can learn its own features without manual feature engineering.

Choosing the number of hidden layers and neurons per layer is called network architecture. A network with multiple layers is sometimes called a **multilayer perceptron** in literature.

### Image recognition example

The input will be an image, say 1000x1000 px. So the input layer will take 1M numbers (the pixels) and pass them on.

What might happen is that neurons in early layers detect small line segments in various orientations. Subsequent layers then might detect higher-order features like eyes, noses, etc. And this continues with subsequent layers detecting more complete face information.

This process happens without human programming or input.

## Neural network model

Layers are the fundamental building blocks of NNs. Suppose you have a layer with 3 neurons that takes 4 input features. Each neuron computes a logistic function. So each neuron has parameters that we denote ((w1,b1),(w2,b2),(w3,b3)). These each compute an activation value (a1,a2,a3).

We often index the layers with the input layer starting at 0. A square bracket superscript is often used to denote a layer index. This superscript is used on weights and activations.

When we say a network has N layers, it has N-1 hidden layers and 1 output layer. The input layer is not included.

For an arbitrary layer `l` and unit `j` the equation is `a_jl = g(w_jl \dot a_lm1) + b_jl`. `g` is the **activation function**. We've used sigmoid but there are others.

## Making inferences with neural networks (forward propagation)

Consider the example of handwritten digit recognition. We use an 8x8 image of 64 pixel values of grayscale values [0,255]. We use a 3 layer network with 25, 15, and 1 unit(s) resp.

We apply the rules above to compute the activation for each layer. Each layer's activation has N elements where N is the number of units in the layer. Sometimes `f(x)` is used to denote the overall output of the model.

This is called **forward propagation**

It is common to have a large number of units in early hidden layers with the number of units decreasing in subsequent layers.

## Inference in TensorFlow

When using TensorFlow you can make layers like `Dense` and specifying the units and activation functions. Then you call that layer like a function to apply it to the input and compute the activation.

At some point you've got to set the parameters, usually obtained from learning.

### Data representation in NumPy and TensorFlow

There are inconsistencies between NumPy and TensorFlow. TensorFlow tends to represent everything as 2D. So just make sure to use a 1xN vector instead of an N vector: `x = np.array([[2.0, 17.0]])`. TensorFlow uses `Tensors` to represent data.

## Building a neural network in TensorFlow

You can use the `Sequential` class to create a sequential neural network that can be worked with.

```python
model = Sequential([
    Dense(units=3, activation='sigmoid')
    Dense(units=1, activation='sigmoid')
    ])

x = np.array([[200,0, 17.0],
             [120.0, 5.0],
             [425.0, 20.0]
             [212.0, 18.0]])
y = np.array([y,0,0,1])
model.compile(...) # inputs TBD
model.fit(x,y)     # train
xnew = np.array(...)
y_predict = model.predict(xnew) # forward propagation / inference
```
## Neural network implementation in Python

Here's an idea of how you'd implement forward propagation for a single layer. The idea is that each layer takes in the previous layer's activation and the current layer's weights and biases.

To do this, create a matrix from the weight vectors with each neruon's weights as a column (matrix is num input features x num units/neurons)

create `b` as a vector of the biases and the input activation as another vector.

## Efficient implementation of NNs w/ vectorization and matrix multiplication

Vectorization and matrix multiplication are very efficient in hardware, parallel hardware, and GPUs. You can implement a step of forward prop like `Z = np.matmul(A_in,W) + B; A_out = g(Z)` where `A_in` is a `1 x num_features` vector, `W` is a `num_features x num_units` matrix, `B` is a `1 x num_units` vector, and `A_out` will be a `1 x num_units` vector.

## Comments on AGI

The course made a few comments on AGI. Namely that Andrew believes it is still far out and he doesn't see a direct path to get there yet.

They mentioned the "one learning algorithm hypothesis". It is backed up by evidence like if you rewire connections in one part of the brain to connect to other input sources, that part of the brain adapts to use the new inputs. For example rewiring the auditory cortex to receive sight info causes that part of the brain to process sight.

Other interesting experiments were shown that do similar things to send various signals to simulate another sense. E.g. sending voltages to the tongue of greyscale images can cause sight, some humans can learn echolocation, a haptic (vibrating) belt can be used to induce directional (N, S, E, W) sense, implanting a third eye on a frog can result in the frog adapting and using it.

Given this, a question is to determine what the general learning algorithm used by the brain is and to see if we can emulate that in computers.


