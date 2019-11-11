# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
import os

import pytest
import linfa_k_means
print(dir(linfa_k_means))
# from linfa_k_means import KMeans


@pytest.fixture(scope="session", autouse=True)
def make_data():
    return make_blobs()


def test_k_means_rust(benchmark, make_data):
    dataset, cluster_index = make_data
    model = KMeans(3)
    labels = model.fit_transform(dataset)
    print(model)
    assert labels == cluster_index
