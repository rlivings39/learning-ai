# Machine Learning on Coursera

Course link: https://www.coursera.org/learn/machine-learning/

See [ai-for-everyone.md](../ai-for-everyone/ai-for-everyone.md) for background info

Machine learning helps solve problems where we struggle to write explicit programs to solve. We can let machine learning learn solutions where explicit solutions are hard or difficult.

## What is machine learning

Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed (Arthur Samuel 1959)

Samuel wrote an ML algorithm to learn to play checkers that learned by playing many games of checkers.

Supervised learning, unsupervised learning, and recommendation systems are the most common. Supervised learning is the one that's made the most progress.

## Supervised learning

Supervised learning learns a mapping (x -> y) by being given many inputs along with the correct output label. Eventually it predicts the mapping on its own.

Examples include spam filtering, speech recognition, machine translation, online advertising, and defect detection

Problems solved include regression (predicting numbers out of infinite possibilities) and classification (predicts one of a finite set of outputs like [is cancer, not cancer], [horse, violin, piano]).

Multiple input values can be used in an ML problem such as patient age and tumor size. In such problems the learning algorithm may compute a boundary curve, surface, etc. between various clusters of points.

## Unsupervised learning

Unsupervised learning is given unlabeled data and works to discover structure or something interesting without human intervention.

Examples are clustering algorithms, anomaly detection (e.g. fraud detection in financial transactions), dimensionality reduction (compress data without losing much info)

For example, the algorithm may perform clustering and show what those clusters are. The algorithm determines the number of clusters and membership in said clusters.

Google news uses clustering by looking at tons of articles and groups them together. Grouping people into genetic clusters is another example. Grouping customers is another use case.

## Linear regression

Predicting the price of a house based on the square footage of the house is a problem amenable to linear regression.

Linear regression is probably the most used machine learning algorithm in the world

Linear regression with one variable (a single feature) is called univariate linear regression

In linear regression, we try to fit a linear function `f_w,b(x) = wx + b` where we need to determine the best values of the parameters `w, b`. All models will have these parameters which need to be determined. They are also called coefficients or weights.

### Cost function

We use a cost function to quantify how well a model/prediction fits training data. For linear regression we compute the squared error `sum((y_hat_i - y_i)^2 / 2m`. The extra division by 2 makes later calculations neater. It is usually denoted `J(w,b)` and called the squared error cost function.

Many other cost functions are used for different applications. Squared error cost function is used for most regression.

Once the cost function is defined, the goal of training/fitting is to find parameters that minimize the cost function. It can be useful to plot or visualize the cost function to see where it may be minimized. This likely won't be possible given sufficiently complex cost functions but optimization techniques can be applied.

For linear regression you have 2 independent parameters so the cost function is a surface. Looking at contour plots of the cost function can give insight into where the minimum is .

## Training a supervised learning model

1. Collect training set of features and targets
2. Feed that training set to the learning algorithm
3. Get the prediction function `f` (older: hypothesis)

The prediction value function maps `x` to the prediction/estimated denoted `y-hat`

### Training with gradient descent

Gradient descent can be used to optimize general functions of multiple variables to minimize the cost function `J(w1, w2, ..., wn, b)`. It is used for many kinds of models including some neural networks.

Process outline

* Start with some initial guess. Sensitivity to this guess varies between models.
* For linear regression, `w=0, b=0` is a fine choice
* Note that there may be multiple minima
* Start at the point and find the direction of steepest gradient
* Step in that direction
* Repeat

Gradient descent has an interesting property where it may settle to a local minimum but not necessarily a global minimum

Take steps looking like

```
w = w - a*d/dw*J(w,b)
b = b - a*d/db*J(w,b)
```

until convergence happens. I.e. `w` and `b` change very little on each step.

alpha is the "learning rate` between 0 and 1 controlling the size of the steps to take. The derivatives are the partial derivatives.

The partial term is subtracted so that you descend for positive slope and ascend for negative slope.

The cost function for linear regression is convex so there's only one local minimum which matches the global minimum.

**Batch gradient descent** Uses all training samples in each step. There are other algorithms using subsets of the data.

### Choice of the learning rate

The choice of the learning rate impacts efficiency and effectiveness of training.

Too small of a choice causes training to be slow.

Too large of a choice can cause training to fail by overshooting, failing to converge, and/or diverging.

With a fixed learning rate, steps decrease as you approach the minimum because derivatives generally decrease near the minimum.

## Multiple linear regression

Similar to linear regression but with multiple input variables. So each `xi` is a vector of the `n` features/variables. These are indexed as `x^i_j` where the superscript `i` is the training example index and the subscript `j` is the index of the feature.

The formula is `fwb(x_vec) = w1*x1 + w2*x2 + ... + wn*xn + b = w_vec \dot x_vec + b`

**Note** Use vectorized operations in NumPy rather than explicit loops for faster execution.

## Alternative: normal equation

Using the normal equation and linear algebra you can solve for `w, b` without iterating. This doesn't generalize to other algorithms and can be slow when the number of features is large.

Some ML libraries may use the normal equation in the backend to solve in the background.

## Feature scaling to make gradient descent work better

Suppose we model a house price in terms w/ parameters of size in sq ft and number of bedrooms. For typical houses the size ranges from 300-2000 or more and bedrooms ranges between 0 and 5.

Having a situation where you have one large value and range feature and another small value and small range feature causes an imbalance. Small changes in the larger feature can result in large changes to the cost whereas larger changes to the smaller feature result in smaller cost changes. I.e. the contour plot will have ellipses with fairly high eccentricity.

With this, gradient descent may bounce around before finding a global minimum because of the extra sensitivity along the large feature dimension.

You can scale your features to have similar ranges to improve gradient descent.

One way to scale features is to simply divide them by the maximum. You can also perform **mean normalization** to center them around zero. To perform this scaling and shifting you can compute `x_normal = (x - x_mean) / (x_max - x_min)`

That'll put the data close to the range `[-1, 1]`.

**z-score** normalization leverages the standard deviation to normalize `x_normal = (x - x_mean) / (x_stddev)` where `x_stddev` is the standard deviation of the feature `x`.

A goal is to rescale so features are close to the range `[-1, 1]`. They don't have to be exact, close is good enough.

There is rarely harm to using feature scaling, so erroring on the side of using it is preferable.

### Checking for gradient descent convergence

Plotting a **learning curve** with the number of iterations on the x axis and cost on the y axis can help you see how the computation is converging.

In addition, you can use an automatic convergence test in your gradient descent algorithm by setting a small threshold. Andrew says that choosing the right threshold can be hard so he often uses learning curves.

### Choosing an appropriate learning rate

**Resource** The [feature scaling and learning rate lab](coursera-ai-courses/machine-learning/week2-labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb) does an excellent job of demonstrating the ideas discussed here.

If the cost is oscillating between increasing and decreasing this could mean you've got a bug in your implementation or that the learning rate is too large. If the learning rate is far too large the cost may just increase forever leading to divergence.

With a small enough learning rate the cost should decrease on every iteration. As a debugging step, just set the learning rate to a tiny number and ensure that the cost decreases. If that fails, you've likely got a bug.

Using a learning rate that's too small can cause slow convergence. Trying a range of values like 0.001, 0.01, 0.1, 1, etc, running gradient descent for a few steps, plot the cost, and see what's reasonable.

Andrew tries successively tripling the learning rate to find values that are too small and too large. Then pick a learning rate that's close to the largest sane value.

When implementing feature scaling, you must store the scaling parameters (e.g. min, max, mean, stddev) so that they can be used during prediction.

## Glossary and notation

* **Training set / training data** Data used to train the model
* **Input variable / input feature** The input to the model (`x`)
* **Output variable / target variable** The output predicted by the model (`y`)
* `m` The number of training examples
* `(x^i, y^i)` (superscripts) The i'th sample

