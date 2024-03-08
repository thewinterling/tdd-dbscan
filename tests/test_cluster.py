#!/usr/bin/env python3
import pytest
from collections.abc import Iterable

from dbscan.cluster import Cluster


def test_cluster_is_empty_when_no_point_was_added():
    cluster = Cluster()
    assert cluster.is_empty


def test_cluster_getitem_raises_exception_when_index_is_out_of_bounds():
    cluster = Cluster()
    with pytest.raises(IndexError):
        _ = cluster[0]


def test_cluster_is_iterable():
    cluster = Cluster()
    assert isinstance(cluster, Iterable)


def test_cluster_holds_indices_to_two_points_when_previously_added():
    cluster = Cluster()

    point_index_1 = 19
    point_index_2 = 8
    cluster.add(point_index_1)
    cluster.add(point_index_2)

    assert len(cluster) == 2
    assert cluster[0] == point_index_1
    assert cluster[1] == point_index_2
