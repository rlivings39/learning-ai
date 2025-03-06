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


