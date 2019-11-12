# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
import os
import numpy as np

import pytest
from linfa_k_means import KMeans
from sklearn.cluster import KMeans as sk_KMeans

@pytest.fixture(scope="session", autouse=True)
def make_data():
    return make_blobs(n_samples=1000000)

def test_k_means_rust(benchmark, make_data):
    dataset, cluster_index = make_data
    model = KMeans(3, max_iter=100)
    labels = benchmark(model.fit_predict, dataset)
    assert len(labels) == len(cluster_index)

def test_k_means_python(benchmark, make_data):
    dataset, cluster_index = make_data
    model = sk_KMeans(3, init="random", algorithm="full", max_iter=100)
    labels = benchmark(model.fit_predict, dataset)
    assert len(labels) == len(cluster_index)

def test_serialisation(make_data):
    dataset, _ = make_data
    model = KMeans(3, max_iter=1)
    model.fit(dataset)
    path = "testpath"
    model.save(path)
    model2 = KMeans.load(path)
    assert (model.cluster_centers_ == model2.cluster_centers_).all()
