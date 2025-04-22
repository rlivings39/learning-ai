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

### What to do next to improve model performance? Diagnosing bias and variance

Looking at the training and cross validation errors can give indicators of bias and variance (i.e. underfitting and overfitting).

* High training error, high cross validation error - indicative of high bias or underfitting
** If cross validation error is much higher than the training error in this case you may wind up with high bias **and** high variance
* Low training error, high cross validation error - indicative of high variance or overfitting
* Low training error, low cross validation error - could be a good fit

When increasing the degree of the polynomial, the training error will tend to decrease. The cross validation error will be concave up with a minimum in the middle.

When using regularization in the fitting process to fit a polynomial, you can compute the regularization parameter (i.e. lambda) using a similar strategy by comparing the training and cross validation errors for various values of lambda.

As lambda increases, the training error will typically increase because training prioritizes minimizing the weights. Cross validation error tends to be concave up with a minimum somewhere in the middle.

### Establishing a baseline level of performance

A good way to evaluate your model's performance is to measure against a baseline performance. Ask yourself the question "What is the best performance I could reasonably expect?". For example you can compare against

* Human level performance
* Competitor performance
* An informed guess

The gap between the baseline performance and training error diagnoses bias. The gap between training error and cross validation error diagnoses variance.

### Learning curves

A learning curve plots error vs. training set size. You can plot both training and cross validation error. Cross validation error tends to decrease as training set size increases. Training error tends to increase as the training set size increases.

This happens because it is harder to fit more data points as the training set increases. With 1 or 2 samples you can often fit perfectly with 0 error.

Cross validation error is usually higher than the training error.

In the case of high bias (underfitting), both the training and cross validation error start out steep and then flatten out after a while. Things flatten out because there are too few parameters to fit more data. Human level performance will tend to be much lower than the two errors.

Throwing more training data at a model with high bias will offer no benefit.

In the case of high variance (overfitting), you'll see a huge gap between the training error and cross validation error. Human level performance may actually be worse than the training error in the case of overfitting. Adding more training data may actually improve the model's performance.

Plotting learning curves is expensive because you have to train the model for each training set size to plot the graph.

### Concrete steps to improve performance

Here are concrete strategies to fix issues.

To address high bias (underfitting):

* Try getting additional features
* Try adding polynomial features
* Try increasing lambda

To address high variance (overfitting):

* Get more training examples
* Try decreasing lambda
* Try smaller sets of features

### Bias and variance in neural networks

Large NNs are low bias machines

If the model does poorly on the training set, try a bigger network (i.e. more hidden layers and/or more units). Repeat until you're happy.

Then see if the model does well on the cross validation set. If not, get more data and go back to the beginning.

This applies well to a limit. Huge networks are expensive to train. There are limits to how much data you can gather.

With proper regularization, larger neural networks will perform no worse than a smaller network. So going bigger rarely hurts the model's performance.

In TensorFlow you add regularization to the layers `Dense(units=25, activation='relu', kernel_regularizer=L2(0.1))`

## Machine learning development process

The overall loop will look like

1. Choose architecture, model, data, etc.
2. Train model
3. Diagnose (bias, variance, error analysis)
4. Change decisions (regularization, learning, add/subtract features) GOTO 1

Fun idea to get more spam email data is to create tons of fake email addresses and leak them to spammers. Then use those emails as input to train the system.

### Error analysis

Second most important thing to analyze after bias and variance.

Manually analyze the failure examples from the cross-validation set and try to group them on common traits. For the spam email example maybe they're pharmaceutical emails, have deliberate misspellings, unusual email routing, stealing passwords, spam message in image. Bucket the emails into categories and see where to focus your effort

Error analysis may not always be easy for humans, e.g. predicting when someone will click an ad. But when errors are analyzable it is quite useful.

### Adding or creating more data for your ML problem

Focus on adding more data of the types identified by error analysis.

You can perform data augmentation / modification to create new training examples from your existing data by applying transformations, distortions, etc. FOr example you might rotate or warp images to produce several. For speech you might add various types of background noise.

**Tip**: Changes or distortions made to the data should be representative of what you expect to happen in the test set or real life prediction data.

Data synthesis is the idea of creating artificial data that could realistically show up in your test/prediction sets. E.g. creating images with text using various fonts and colors on your computer.

Given that ML algorithms are fairly mature today, taking a data-centric approach can be an efficient way to improve your algorithm's performance.

### Transfer learning: using data from a different task

Suppose you're trying to train a system to match digits 0,1,2,...,9. You do have data for classifying images into 1000 categories like cat, dog, car, person, .... You train that model and then remove the output layer and replace it with a different layer with 10 units outputting a match to your digits.

The parameters from the old network can't transfer over for the new output layer. You can reuse the hidden layers and then do one of two things. Option 1: train only the output layers. Option 2: train all the parameters in the network initializing the parameters from the other model.

The fine tuning training step can then often be effective with much smaller training sets.

This algorithm is called transfer learning because one supposes that learning something related can transfer to a new task.

There may be freely trained networks available for download that you can fine tune for your task without having to spend time on pretraining.

An idea behind why transfer learning can help is that similar tasks of have similar basic steps. For example, with image recognition preliminary learning on how to identify edges, corners, curves, etc. will be useful to all similar tasks.

### Full cycle of a machine learning project

1. Define project scope
2. Define and collect data
3. Train model, error analysis, iterative improvement
4. Deploy in production, monitor, maintain system, leverage live data for more error analysis and training improvements

Deployment typically means having a server performing inference by running predictions with your model. This will be exposed to users in some way via say a mobile app or website. The app or website sends an API request to the inference server to carry out predictions and sends the results back.

Software engineering may be needed to

* Ensure reliable and efficient predictions
* Scaling
* Logging (privacy is important here)
* System monitoring
* Model updates after retraining, etc.

### Fairness, bias, and ethics

Failures exist in the past showing bias including a hiring tool that discriminated against women and a facial recognition system matching dark skinned people to criminal mug shots. Biased bank loan approvals happened. Models can also perpetuate toxic stereotypes. Such systems need to be avoided.

There are also adverse use cases such as deep fakes showing a fake video of Barack Obama.

Ideas to improve things:

Get a diverse team to brainstorm things that might go wrong with emphasis on possible harm to vulnerable groups. Andrew mentions that diversity here can be important to identify potential issues.

Carry out a literature search for standards and guidelines for your industry.

Audit systems against possible harm prior to deployment. After you train but before deployment, measure the performance to see if any potential identified bias exists.

Develop a mitigation plan (e.g. rollback to a less biased system) and monitor for possible harm.

## Error metrics for skewed datasets

Traditional error metrics can be very wrong when working with skewed datasets (i.e. where the vast majority of cases are in a single class). For example suppose you have a rare disease where only 0.5% of patients have it. Then a simple algorithm like `y = 0` will have 99.5% accuracy, 0.5% error but will be terrible.

**Precision** and **recall** are often used as metrics. Suppose y = 1 is the rare class we want to detect. Compute the **confusion matrix** as pairs of the actual class and predicted class. In the case of actual class and predicted class being 1 we get a **true positive**. With both classes 0 we get a **true negative**. With actual 0 and predicted 1 we get a **false positive** and with actual 1 and predicted 0 we get a **false negative**.

```
Precision = # true positives / # predicted positives = true positives / (true positives + false positives)

Recall = true positives / # actual positives = true positives / (true positives + false negatives)
```

There is often a trade off between precision and recall. Consider an example of using logistic regression. If you increase the threshold you increase the precision but decrease the recall because you minimize the number of false positives but increase your false negatives.

Plotting precision vs. recall for different threshold values can help you choose a good point.

The **F1 score** can help automatically choose a balance between precision and recall. The F1 combines precision and recall but penalizes small values more harshly (this is the **harmonic mean**)

```
F1 = 1/0.5*(1/P + 1/R) = 2*PR/(P+R)
```

## Decision trees and tree ensembles

Decision trees are widely used and often successful. Consider a cat classification example for an adoption center. We have features like ear shape, face shape, whiskers, and a boolean of whether the animal is a cat.

The tree works by arranging the features as nodes in the tree with categorical values of each node as the outgoing edges. Eventually you can make an inference based on this. The **root node** is the top node. Internal nodes are **decision nodes**. The prediction nodes are **leaf nodes**.

### Decision tree learning

The general algorithm looks something like

1. Decide which feature will be the root node. Training examples are split based on that.
2. Then decide which feature to put on each outgoing edge of the root node.
3. Continue until you hit a leaf node with all matching classifications.

The **purity** of a collection of examples is used to determine the quality of a decision. The **entropy** is used to measure impurity. Taking `p1` to be the proportion of positive examples labeled 1 is `H(p1)` which is something like an upside down parabola with vertex at 0.5 that intersects the x axis at 0 and 1. So we prefer values close to 0 or 1 and penalize values close to 0.5 as those are undesirable.

For binary classification we have `p1, p0 = 1 - p1`. Then we define

$$
\begin{align*}
H(p_1) &= -p_1 * log_2(p_1) - p_0 * log_2(p_0) \\
       &= -p_1 * log_2(p_1) - (1-p_1) * log_2(1 - p_1)
\end{align*}
$$

with the assumption that $H(0) = H(1) = 0$ since $log(0)$ is undefined.

There are other criteria similar to this entropy criteria that can also be used.

### Choosing features to split on: information gain

When choosing which feature to split on, we seek reduced entropy or **information gain**. For each possible feature, compute the entropy for each branch and then compare across features with a weighted average based on the branch sample size and see how much we've reduced that entropy compared to the parent node.

Compute

$$
H(p_1^{parent}) - \frac{m1}{N} * H(p_1^{left}) + \frac{m2}{N} * H(p_1^{right})
$$

for each feature. This computes the **information gain** of each decision. This information gain can also be used as a stopping criterion for the learning algorithm.

### Decision tree learning

1. Start with all examples at the root
2. Calculate information gain for all possible features, choose the one with the highest information gain
3. Split the dataset on the selected feature and branch the tree
4. Repeat until stopping criteria is met
   * When a node is 100% one class
   * When splitting reaches tree max depth. This can help avoid overfitting.
   * Information gain from splits is below threshold
   * Number of examples in a node is below threshold

### Handling features w/ more than 2 possible values: one-hot encoding

Features with more than 2 values we can use multiple children in your tree.

Alternatively this can be solved by splitting your N-ary feature in N binary features where 1 means that category is true and 0 otherwise. Exactly one of the N features will take the value 1.

This transformation is called a **one hot encoding**. Such an encoding can be used for other learning algorithms which expect numbers as inputs.

### Decision trees with continuous valued features

Given a continuous feature, split based on a threshold. A good value for the threshold can be chosen by sorting the training examples on the feature value, then use the midpoints between consecutive samples as potential thresholds. Choose the one with the best information gain.

### Regression trees

Decision trees can be generalized to predict continuous features (i.e. regression rather than classification).

The leaf nodes in the decision tree predict the average of the members of samples reaching that leaf node.

When choosing a feature to split we try to minimize the variance of the output variable. Compute the weighted mean of the variance of each bucket and subtract from the variance of the parent. Choose the one which maximizes this reduction in variance:

$$
\sigma^2_{parent} - \left(\frac{m_1}{N} * \sigma^2 + \frac{m_2}{N} * \sigma^2\right)
$$

### Tree ensembles

A weakness of using a decision tree is that the single tree can be highly sensitive to small changes in the data. For example, changing a single training example changes the entire tree structure.

More accurate predictions can result when you train multiple trees. With multiple trees, you use the majority vote to get the final prediction. Having multiple trees makes your algorithm more robust.

To generate a tree ensemble given a training set of size `m`

1. Use sampling with replacement to create `B` training sets of size `m`
2. Train a decision tree on each of the `B` training sets

Values of `B` are commonly like 32, 64, or 100. Andrew says that using more than 100 is not usually helpful. This is called a **bagged decision tree**

Even with this sampling with replacement procedure, you may often wind up with the same root node splits and other nodes close to the root.

Another modification is when choosing a feature to split on if you have `n` features to choose from, pick `k < n` features to split on with `k = sqrt(n)` and only split on a feature from that subset. This is the **random forest algorithm** that can help make your tree ensembles more robust because you've already baked in some perturbations.

### Boosted trees and XGBoost

To generate a tree ensemble given a training set of size `m`

1. Use sampling with replacement to create `B` training sets of size `m`. When sampling make it more likely to choose examples that were previously miscategorized. This is the idea of *deliberate practice* where you practice things that you're bad at rather than things you're good at (e.g. learning an instrument).
2. Train a decision tree on each of the `B` training sets
3. Use the current ensemble to predict on each of the training examples and record the results to be used in the next iteration

 The details on how to choose the weights when sampling are complicated. XGBoost (eXtreme Gradient Boost) is a common, fast, effective implementation. It has good default sampling criteria and stopping criteria as well as built-in regularization. It's highly competitive for ML competitions like Kaggle.

 XGBoost and deep learning seem to win most of the competitions.

 There are many libraries implementing XGBoost, like `xgboost` in Python. `XGBClassifier, XGBRegressor` in that library implement those algorithms.

 ### Practicalities around decision trees and tree ensembles

 Tuning the hyperparameters for decision trees like minimum sample size to split, max tree depth, and number of trees in your ensemble can have massive impacts on your learning algorithm. [C2_W4_Lab_02_Tree_Ensemble.ipynb](coursera-ai-courses/advanced-learning-algorithms/week4-labs/C2_W4_Lab_02_Tree_Ensemble.ipynb) is a great example of this.

 `sklearn` has `GridSearchCV` which can help do this search and finally fit and predict with the best example.

 ### When to use decision trees

 Decision trees and neural networks are powerful with tradeoffs

 Decision trees and tree ensembles

 * Work well on tabular (structured) data. E.g. if your data looks like a big spreadsheet w/ categorical or continuous data
 * Not recommended for unstructured data like images, audio, text
 * Fast to train and evaluate
 * Small decision trees are human interpretable. This falls off with larger trees or ensembles.
 * XGBoost is recommended as a default strategy

Neural networks

* Work well on all types of data including tabular (structured) and unstructured data. They are preferred for unstructured data
* May be slower than a decision tree, especially for learning
* Works well with transfer learning to effectively train with smaller training sets
* When building a system of multiple models working together, it can be easier to string together multiple neural networks by training them all together using gradient descent
