# XGBoost Tree Regression Based on [Video](https://www.youtube.com/watch?v=OtD8wVaFm6E&ab_channel=StatQuestwithJoshStarmer)

- XGBoost Trees (different from our regular trees)
- Most common Regression for XGBoost:
  ### First Epoch
  - Initial prediction is made (by default 0.5)
  - Each tree starts as single leaf
  - Similarity score of residuals: (sum of residuals)^2 / (no. of residuals + lambda)
  - lambda: regularization param (starts with 0)
  ### Subsequent Epochs
  - Take two datapoints (start with extreme label vals and move towards other end), find the avg
  - Split the data based on the average of label vals and find simliarity of left, and right
  - Find Gain using formula: Left_similarity + Right_similarity - Root_similarity
  - The avg of the two datapoints which results in the largest gain will be used as the split.
  - We continue to do this split for multiple levels deep (default max depth is 6).
    #### Tree Pruning
    - We keep a gamma value and on each branch of the (starting from the bottom most) tree, we compare the gain with gamma. If (gain - gamma < 0) on a branch, prune that branch.
    > Side NOTE: 
    >- A lambda (regularization) helps removing the outliers in the data, preventing overfitting. It decreases the gain value on each branch.
    >- Regularization also helps in decreasing the similarity score. And the amount of decrease in scores is inversely proportional to the number of residuals in the node.
    >- In the video example, if lambda is 1 and using a gamma even of value 0 leads to pruning of the tree branch since the gain becomes negative. That means when lambda
  - Output value for the leaf nodes is calculated using formula (`sum of all residuals / (number of residuals + lambda)`).
  - Learning rate (default 0.3) is applied to each leaf node and values are updated from the predicted value (`leaf_residual = prediced_val + eta x leaf_output`).

# XGBoost Tree Classification
- Start with a prediction probability of 0.5.
- In each epoch, 