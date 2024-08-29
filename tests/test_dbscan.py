#!/usr/bin/env python3

import pytest
import numpy as np
import sklearn.cluster as baseline

from dbscan import dbscan
from dbscan.point import Point


@pytest.fixture
def random_points_to_cluster():
    """Generates hundred random points in the range 0 < x/y < 50."""
    np.random.seed(0)
    random_points = np.random.rand(100, 2) * 50
    points = [Point(x, y) for x, y in random_points]
    return points


@pytest.fixture
def sample_points_to_cluster():
    """Generates seven explicitly set points."""
    return [
        Point(1, 1),
        Point(1.1, 1.1),
        Point(3.5, 5),
        Point(3.5, 10),
        Point(3.5, 12),
        Point(1.2, 1.2),
        Point(5, 5),
        Point(10, 1.5),
    ]


def test_instantiate_and_config_dbscan():
    epsilon = 0.1234
    minimal_points_per_cluster = 3
    clustering = dbscan.Dbscan(epsilon, minimal_points_per_cluster)
    config = clustering.config

    assert epsilon == config["epsilon"]
    assert minimal_points_per_cluster == config["minimal_points_per_cluster"]


def test_all_points_were_visited(random_points_to_cluster):
    epsilon = 1.5
    minimal_points_per_cluster = 3
    clustering = dbscan.Dbscan(epsilon, minimal_points_per_cluster)
    clustering.run(random_points_to_cluster)

    processed = clustering.clustering_internals["points_processed"]
    assert all(processed)


def test_all_points_are_noise_if_epsilon_is_too_small(sample_points_to_cluster):
    epsilon = 0.1
    minimal_points_per_cluster = 3
    clustering = dbscan.Dbscan(epsilon, minimal_points_per_cluster)
    clustering.run(sample_points_to_cluster)

    noise_points = clustering.clustering_internals["noise_points"]
    assert all(noise_points)


def test_three_points_should_be_in_one_cluster(sample_points_to_cluster):
    epsilon = 1
    minimal_points_per_cluster = 3
    clustering = dbscan.Dbscan(epsilon, minimal_points_per_cluster)
    clusters = clustering.run(sample_points_to_cluster)

    noise_points = clustering.clustering_internals["noise_points"]
    core_points = clustering.clustering_internals["core_points"]
    number_of_found_clusters = clustering.clustering_internals["number_of_found_clusters"]
    assert not noise_points[0]
    assert not noise_points[1]
    assert not noise_points[5]

    assert noise_points[2]
    assert noise_points[3]
    assert noise_points[4]
    assert noise_points[6]
    assert 1 == number_of_found_clusters and 1 == len(clusters)
    assert 3 == len(clusters[0])
    # Checks that a point can never be a core _and_ a noise point.
    assert not any([c & n for c, n in zip(core_points, noise_points)])


def test_compare_results_to_sklearn_dbscan(sample_points_to_cluster):
    epsilon = 1.0
    minimal_points_per_cluster = 3
    baseline_dbscan = baseline.DBSCAN(eps=epsilon, min_samples=minimal_points_per_cluster)
    sample_points_to_cluster_numpy = np.array([[p.x, p.y] for p in list(sample_points_to_cluster)])
    baseline_clusters = baseline_dbscan.fit(sample_points_to_cluster_numpy)

    our_clustering = dbscan.Dbscan(epsilon, minimal_points_per_cluster)
    _ = our_clustering.run(sample_points_to_cluster)
    clusters = our_clustering.clusters

    for index, baseline_cluster_id in enumerate(baseline_clusters.labels_):
        our_cluster = clusters[baseline_cluster_id]
        if baseline_cluster_id != -1:
            assert index in our_cluster
        else:
            assert len(our_cluster) == 0
