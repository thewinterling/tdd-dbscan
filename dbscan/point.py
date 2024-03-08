#!/usr/bin/env python3

import numpy as np


class Point(object):
    def __init__(self, x: float, y: float):
        """A point class that can be used within the dbscan cluster algorithm
        and implements a `is_in_neighborhood` method that defines when
        two instances of Point are neighbors.

        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.
        """
        self.x = x
        self.y = y

    def __str__(self):
        return self.__repr()

    def __repr__(self):
        return f"Point: x: {self.x} , y: {self.y}"

    def is_in_neighborhood(self, other: "Point", epsilon: float) -> bool:
        """Calculates if the `other` point is in the neighborhood of the
        one at hand.

        Args:
            other (Point): Second instance of a Point.
            epsilon (float): Epsilon value that defines the actual neighborhood.
        Returns:
            bool: True if both points are close enough to be considered
                in a neighborhood, False otherwise.
        """
        distance = np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        return distance < epsilon
