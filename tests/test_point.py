#!/usr/bin/env python3

import numpy as np

from dbscan.point import Point


def test_point_values_are_correctly_set():
    expected_x = 0.1
    expected_y = 1.8
    my_point = Point(expected_x, expected_y)

    assert expected_x == my_point.x
    assert expected_y == my_point.y


def test_point_is_not_in_neighborhood_when_distance_is_bigger_than_epsilon():
    point_1 = Point(0.0, 1.0)
    point_2 = Point(1.1, 0.5)
    epsilon = 1.0

    assert not point_1.is_in_neighborhood(point_2, epsilon)


def test_point_is_not_in_neighborhood_when_distance_equals_epsilon():
    point_1 = Point(0.0, 0.0)
    point_2 = Point(0.5, 0.5)
    epsilon = np.sqrt(0.5)

    assert not point_1.is_in_neighborhood(point_2, epsilon)


def test_point_is_in_neighborhood_when_distance_is_smaller_than_epsilon():
    point_1 = Point(0.0, 0.0)
    point_2 = Point(0.5, 0.5)
    epsilon = np.sqrt(0.5) + 0.1

    assert point_1.is_in_neighborhood(point_2, epsilon)
