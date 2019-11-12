from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms
import numpy as np
from linfa_k_means import WrappedKMeans


class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Means clustering, using a Rust's ndarray instead of Numpy
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence
    random_state : int or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers. TODO? If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    labels_ :
        Labels of each point
    inertia_ : TODO float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : TODO int
        Number of iterations run.
    Examples
    --------
    >>> from linfa_k_means import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    Notes
    -----
    TODO
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):

        self.model_ = WrappedKMeans(random_state, tol, max_iter)
        self.n_clusters = n_clusters
        self.cluster_centers_ = None

    def fit(self, X, y=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        y : Ignored
            not used, present here for API consistency by convention.
        """
        self.model_.fit(self.n_clusters, X)
        self.cluster_centers_ = self.model_.centroids()
        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        y : Ignored
            not used, present here for API consistency by convention.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).predict(X)

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        y : Ignored
            not used, present here for API consistency by convention.
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        return self.fit(X)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'cluster_centers_')

        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return self.model_.predict(X)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')
        return self.model_.predict(X)

    def save(self, path: str):
        self.model_.save(path)

    @classmethod
    def load(cls, path: str) -> "KMeans":
        model_ = WrappedKMeans.load(path)
        cluster_centers_ = model_.centroids()
        model = super().__new__(cls)
        model.model_ = model_
        model.cluster_centers_ = cluster_centers_
        return model

    def score(self, X, y=None):
        """Opposite of the value of X on the K-means objective.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        y : Ignored
            not used, present here for API consistency by convention.
        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self, 'cluster_centers_')

        x_squared_norms = row_norms(X, squared=True)
        return -_labels_inertia(X, x_squared_norms,
                                self.cluster_centers_)[1]


def _labels_inertia(X, x_squared_norms, centers):
    labels, distances = pairwise_distances_argmin_min(
        X=X, Y=centers, metric='euclidean', metric_kwargs={'squared': True})
    labels = labels.astype(np.int32, copy=False)
    inertia = distances.sum()
    return labels, inertia
