Version 0.2.0 - 2020-11-26
==========================

New algorithms
-----------

 - Ordinary Linear Regression has been added to `linfa-linear` by [@Nimpruda] and [@paulkoerbitz]
 - Generalized Linear Models has been added to `linfa-linear` by [VasanthakumarV]
 - Linear decision trees were added to `linfa-trees` by [@mossbanay]
 - Fast independent component analysis (ICA) has been added to `linfa-ica` by [@VasanthakumarV]
 - Principal Component Analysis and Diffusion Maps have been added to `linfa-reduction` by [@bytesnake]
 - Support Vector Machines has been added to `linfa-svm` by [@bytesnake]
 - Logistic regression has been added to `linfa-logistic` by [@paulkoerbitz]
 - Hierarchical agglomerative clustering has been added to `linfa-hierarchical` by [@bytesnake]
 - Gaussian Mixture Models has been added to `linfa-clustering` by [@relf]

Changes
----------

 - Common metrics for classification and regression have been added
 - A new dataset interface simplifies the work with targets and labels
 - New traits for `Transformer`, `Fit` and `IncrementalFit` standardizes the interface
 - Switched to Github Actions for better integration

Version 0.1.3
===========================

New algorithms 
------------

 - The `DBSCAN` clustering algorithm has been added to `linfa-clustering` ([#12](https://github.com/LukeMathWalker/linfa/pull/12) by [@xd009642])
   
Version 0.1.2 (2019-11-25)
===========================

New algorithms 
------------

 - First release of `linfa-clustering:v0.1.0` with the `KMeans` algorithm (by [@LukeMathWalker])
 - First (real) release of `linfa`, re-exporting `linfa-clustering` (by [@LukeMathWalker])
 

[@LukeMathWalker]: https://github.com/LukeMathWalker
[@xd009642]: https://github.com/xd009642
