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

## Training the NN

After you configure the network, call `compile` specifying a loss function, then call `fit` with the training data and number of epochs to use.

The general framework for training a model is:

1. Specify how to compute the output given an input
2. Specify loss function and cost function,
3. Minimize the cost function using gradient descent with `w = w - alpha * dj_dw, b = b - alpha * dj_db`

Neural networks work with the same 3 steps. The loss function is `L(f(x),y) = -y*log(f(x)) - (1-y)*log(1-f(x))`, the same as the loss function for logistic regression. This is called **logistic loss** or **binary cross entropy**.

To solve a regression problem, use another loss function such as `tensorflow.kers.losses.MeanSquaredError`.

**Back propagation** is used to compute derivatives for gradient descent. TensorFlow can use algorithms other than gradient descent like Adam to optimize.

### Back propagation

Back propagation essentially decomposes the computation of a function's output into a computational graph showing the composition of different operations.

Break down the calculation starting from `w` in the formula and build up the graph using the order of operations.

That graph is computed left to right for forward propagation. To compute the derivative work right to left and apply the chain rule to compute the overall derivative. When used this way for a network with N units and P parameters, you can compute derivatives in O(N + P) time rather than O(N*P) as would happen if done naively.

This enables a strategy called automatic differentiation which removes the older need to manually compute and specify derivatives.

TODO How are the derivatives computed at each step of back prop?

## Activation functions and alternatives to sigmoid activation

Commonly used activations are

1. The sigmoid activation function for modeling binary features `g(z) = 1/(1+exp(-z))`
2. Another common activation function is ReLU (rectified linear unit) `g(z) = max(0,z)`
3. A linear activation function is `g(z) = z` is sometimes said to be using "no activation function"
4. Softmax used to choose between several output categories

Convolutional layers are another type of layer. They work by having each unit only analyze a subset of the input data. This allows for faster processing and can help avoid overfitting.

Convolutional layers are parameterized by the window size and overlap of the windows. Effectively choosing these parameters allow you to optimize performance.

### Choosing between activation functions

To choose the output layer activation function

1. If you're solving a binary classification problem, use sigmoid
2. If `y` can be positive or negative, use linear
3. If `y` can only be nonnegative, use ReLU

For hidden layers ReLU is the most common choice. Originally, sigmoid was used much more frequently for hidden layers.

ReLU is faster to compute. ReLU is only flat in 1 part of the graph meaning that gradient descent is only flat (i.e. slow) in one place. So, ReLU can help speed up learning.

There are other activation functions like leaky ReLU and others which may yield some benefit in certain cases so just be aware of them

### Why do we need activation functions at all?

Can't we just use linear activations (aka no activations)? If you did that, the network effectively regresses to linear regression and yields no more power. This derives from the fact that the composition of N linear functions is again a linear function.

If all hidden layers are linear and the output layer is sigmoid, then that is equivalent to logistic regression.

Heuristic: don't use linear activations for hidden layers and just start with ReLU

## Multiclass classification with softmax

This is the set of problems with more than 2 output classes such as classifying all written digits or identifying which of 5 diseases a patient has or determining one of many types of defects.

The output of logistic regression is thought of as the probability of `y` being 1. So it also computes the probability of `y` being equal to 0 which is `1 - P(y=1|x)`

If you have N classes with weights and biases for them compute `z_i = w_i \dot + b_i` for `i=1,2,...,N`. Then compute the probabilities `a_i = exp(z_i) / sum(exp(z_k))`. Each `a_i` is the probability that `y` is equal to `i`. We always have `sum(a_i) == 1`.

With N=2, the computation reduces to basically logistic regression.

The loss for softmax regression is `loss(a1,...,an,y) = -log(a_i) for y = i`

To leverage this in a NN, use the same hidden layers and an output layer that uses softmax

For TensorFlow you can use the 'softmax' activation for your output layer and the `SparseCategoricalCrossentropy` loss

### More numerically stable computation of logistic and softmax loss

For binary classification, instead of using a sigmoid output layer with `BinaryCrossEntropy` for the loss in `fit` use a `linear` output layer and use `loss=BinaryCrossEntropy(from_logits=True)` which allows for more numerical stability.

For logistic regression, this isn't too big of a deal.

Do the same thing for softmax where you use a linear output layer and `loss=SparseCategoricalCrossEntropy(from_logits=True)`

What this means is that the intermediate terms passed to `log` are not directly precomputed but substituted in place.

**Note** With this modification, you now have a linear layer as the output so you must pass the output through `tf.nn.sigmoid` or softmax to get the final prediction probability.

### Multi-label classification

Here, the output is a vector of booleans for each label. You can build N networks to detect each label. You can also build 1 network to do all of them at once and have an output layer that uses N sigmoid activations.

The [multiclass lab](./week2-labs/C2_W2_Multiclass_TF.ipynb) demonstrates this and has a nice breakdown of how each unit in the NN's layers segments the space in order to perform the classification.

## Advanced optimization algorithms (gradient descent alternatives)

Recall that the update step in gradient descent is `wj = wj - alpha * d/dwjJ(w,b)`

An idea is to also vary the learning rate during optimization time. If you take many steps in one direction, increase the learning rate to accelerate convergence. If the steps being taken are all over the place, then decrease the learning rate.

The Adam algorithm (ADAptive Moment algorithm) has a learning rate for each weight and bias and varies them during the optimization process.

To use a different optimizer, pass the `optimizer=tf.keras.optimizers.Adam(learning_rate=val)` argument to `model.compile`. You can tune the initial learning rate to get more efficiency.

Adam is the de facto standard for learning today. It's usually a great place to start.

## Comments on AGI

The course made a few comments on AGI. Namely that Andrew believes it is still far out and he doesn't see a direct path to get there yet.

They mentioned the "one learning algorithm hypothesis". It is backed up by evidence like if you rewire connections in one part of the brain to connect to other input sources, that part of the brain adapts to use the new inputs. For example rewiring the auditory cortex to receive sight info causes that part of the brain to process sight.

Other interesting experiments were shown that do similar things to send various signals to simulate another sense. E.g. sending voltages to the tongue of greyscale images can cause sight, some humans can learn echolocation, a haptic (vibrating) belt can be used to induce directional (N, S, E, W) sense, implanting a third eye on a frog can result in the frog adapting and using it.

Given this, a question is to determine what the general learning algorithm used by the brain is and to see if we can emulate that in computers.

## Advice on how to build machine learning models

### Evaluating a model's performance

Evaluate the performance of your model by splitting your training set into 2 parts, say 70% as training data and 30% as a test set or use 80/20.

For regression

Do your usual fit. Then compute a test error, similar to the cost function used in fitting, say the mean squared error, without a regularization term. That will give yo ua sense for how well the model is doing.

You can also compute the training error by checking the error on the training examples. This should be very close to 0 because that's what you trained on.

For classification, you can also simply compute what fraction of the test/training sets have been misclassified.

### Automatically choosing a good model

Suppose you're trying linear regression. You can start by fitting a first order polynomial and compute test error, do the same for a second order polynomial and compute test error, etc. up through a higher order polynomial. Then decide which fits best. **Note** Andrew says this isn't best because you're implicitly fitting the polynomial degree with the test error and can still wind up overfitting.

To address this split your data into 3 sets: training, cross validation or dev set, and test. Maybe 60%, 20%, 20%. You then evaluate the parameters on your cross validation set and look at the model with the lowest cross validation error. Then you can test the generalization on the chosen polynomial degree using the test set. This keeps the test set out of all fitting and makes it a fair evaluation of the model's generalization.

You can do a similar thing for choosing a neural network architecture.

You should make no decisions using your test set as doing so would mean you're implicitly including it in your training process.

The [model evaluation and selection lab](./week3-labs/C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb) is an excellent demonstration of this.
