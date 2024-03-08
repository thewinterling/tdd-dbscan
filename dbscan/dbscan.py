#!/usr/bin/env python3

from typing import List

from dbscan.cluster import Cluster
from dbscan.point import Point


class Dbscan(object):
    def __init__(self, epsilon: float, minimal_points_per_cluster: int):
        """Class for the actual dbscan clustering algorithm.
        
        Args:
            epsilon (float): Epsilon defining the neighborhood of points.
            minimal_points_per_cluster (int): Number of points a cluster needs to hold to be considered an actual cluster."""
        self._epsilon = epsilon
        self._minimal_points_per_cluster = minimal_points_per_cluster

        self._input_points = None

        self._points_visited = None
        self._noise_points = None
        self._assigned_to_cluster = None
        self._clusters = None
        self._current_cluster_index = None

    def _reset_internal_members(self, points_to_cluster):
        self._input_points = points_to_cluster

        self._points_visited = [False for _ in points_to_cluster]
        self._noise_points = [False for _ in points_to_cluster]
        self._core_points = [False for _ in points_to_cluster]
        self._assigned_to_cluster = [False for _ in points_to_cluster]
        self._clusters = [Cluster() for _ in points_to_cluster]
        self._current_cluster_index = -1

    def run(self, points_to_cluster: List[Point]) -> List[Cluster]:
        """Method to perform the cluster algorithm.
        
        Args:
            points_to_cluster (List[Point]): List of points that should be clustered.
        Returns:
            List[Cluster]: A list of all found clusters.
        """
        self._reset_internal_members(points_to_cluster)

        for index in range(len(self._input_points)):
            if self._points_visited[index]:
                continue
            self._points_visited[index] = True

            neighborhood = self.region_query(index)
            if len(neighborhood) < self._minimal_points_per_cluster:
                self._noise_points[index] = True
            else:
                self._current_cluster_index += 1
                self.expand_cluster(index, neighborhood, Cluster())

        return [x for x in self._clusters if not x.is_empty]

    def region_query(self, index):
        point_to_check = self._input_points[index]
        local_neighborhood = list()
        for index, point in enumerate(self._input_points):
            if point_to_check.is_in_neighborhood(point, self._epsilon):
                local_neighborhood.append(index)
        return local_neighborhood

    def expand_cluster(self, point_index, neighborhood, new_cluster):
        new_cluster.add(point_index)
        self._assigned_to_cluster[point_index] = True

        expand_index = 0
        while True:
            if expand_index == len(neighborhood):
                break
            # Use a while loop and this indirection to not loop over a
            # growing container.
            point_index = neighborhood[expand_index]
            if not self._points_visited[point_index]:
                self._points_visited[point_index] = True
                new_neighborhood = self.region_query(point_index)
                if len(new_neighborhood) >= self._minimal_points_per_cluster:
                    neighborhood += new_neighborhood
                    self._core_points[point_index] = True
            if not self._assigned_to_cluster[point_index]:
                new_cluster.add(point_index)
                self._noise_points[point_index] = False
                self._assigned_to_cluster[point_index] = True
            expand_index += 1
        self._clusters[self._current_cluster_index] = new_cluster

    @property
    def clustering_internals(self):
        return dict(
            points_processed=self._points_visited,
            noise_points=self._noise_points,
            core_points=self._core_points,
            number_of_found_clusters=self._current_cluster_index + 1,
        )

    @property
    def config(self):
        return dict(
            epsilon=self._epsilon,
            minimal_points_per_cluster=self._minimal_points_per_cluster,
        )

    @property
    def clusters(self):
        return self._clusters
