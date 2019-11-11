# -*- coding: utf-8 -*-
import os

import pytest
from linfa_k_means import k_means
from sklearn.datasets import make_blobs


@pytest.fixture(scope="session", autouse=True)
def make_data():
    dataset, cluster_index = make_blobs()
    return [list(row) for row in dataset], list(cluster_index)


def test_k_means_rust(benchmark, make_data):
    dataset, cluster_index = make_data
    cluster_index_hat = k_means(3, dataset, 1e-3, 10)
    assert cluster_index_hat == cluster_index
