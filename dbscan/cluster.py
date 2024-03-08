#!/usr/bin/env python3

from typing import Any

class Cluster(object):
    def __init__(self):
        """Cluster class for the dbscan algorithm that holds
        the point indices."""
        self._cluster = list()
        self._iteration_index = -1

    def __len__(self):
        return len(self._cluster)

    def __str__(self):
        return self.__repr()

    def __repr__(self):
        if len(self) == 0:
            return "Cluster: (empty)"
        return f"Cluster: {self._cluster}"

    def __getitem__(self, index):
        return self._cluster[index]

    def __iter__(self):
        return self

    def __next__(self):
        self._iteration_index += 1
        if self._iteration_index < len(self._cluster):
            return self._cluster[self._iteration_index]
        raise StopIteration

    def add(self, point_index: Any):
        """Adds a new point index to the internal list.
        Note that `point_index` is not bound to a particular type.
        
        Args:
            point_index (Any): The point index that should be added.
        """
        self._cluster.append(point_index)

    @property
    def is_empty(self) -> bool:
        """Property to check whether the cluster is empty or not.
        
        Returns:
            bool: True if the interal list is empty, else False.
        """
        return len(self._cluster) == 0
