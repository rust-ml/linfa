Release 0.5.0

Linfa's 0.5.0 release adds initial support for the OPTICS algorithm, multinomials logistic regression, and the family of nearest neighbor algorithms. Furthermore, we have improved documentation and introduced hyperparameter checking to all algorithms.

## Nearest Neighbors

You can now choose from a growing list of NN implementations. The family provides efficient distance metrics to KMeans, DBSCAN etc. The example shows how to use KDTree nearest neighbor to find all the points in a set of observations that are within a certain range of a candidate point.
```rust
// create a KDTree index consisting of all the points in the observations, using Euclidean distance
let kdtree = CommonNearestNeighbour::KdTree.from_batch(observations, L2Dist).unwrap();
let candidate = observations.row(2);
let points = kdtree.within_range(candidate.view(), range).unwrap();
```

## Multinomial logistic regression

Logistic regression models problems with two possible discrete outcomes. Extending this to the multinomial distribution yields the Multinomial Logistic Regression which can model datasets with an arbitrary number of outcomes. 

We will try to model the winequality of 1440 samples, you can find the full example [here]()
```rust
    // fit a Logistic regression model with 150 max iterations
   let model = MultiLogisticRegression::default()
       .max_iterations(50)
       .fit(&train)
       .unwrap();
```
