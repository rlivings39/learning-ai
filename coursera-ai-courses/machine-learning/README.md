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

## Glossary and notation

* **Training set / training data** Data used to train the model
* **Input variable / input feature** The input to the model (`x`)
* **Output variable / target variable** The output predicted by the model (`y`)
* `m` The number of training examples
* `(x^i, y^i)` (superscripts) The i'th sample

