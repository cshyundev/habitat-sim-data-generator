"""A compact median-split BVH over triangles, flattened for GPU traversal.

Brute-force ray-triangle testing is ~O(rays x triangles) and loses badly to
habitat/Bullet's BVH (measured ~4000x more work for a LiDAR sweep). This builds a
bounding-volume hierarchy on the CPU (numpy) and flattens it into flat arrays that
a Metal kernel can traverse with a small per-thread stack (see ``mlx_backend``).

Layout (one entry per node, root = index 0):
    node_min/node_max : float32[Nn, 3]  -- node AABB
    node_left         : int32[Nn]       -- left child index  (internal nodes)
    node_right        : int32[Nn]       -- right child index (internal nodes)
    node_start        : int32[Nn]       -- first triangle in `order` (leaves)
    node_count        : int32[Nn]       -- triangle count; 0 == internal node

``order`` is a permutation of triangle indices; leaves reference contiguous spans
of it, so the caller reorders its triangle data by ``order`` once at build time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BVH:
    """Flattened bounding-volume hierarchy arrays for traversal."""

    node_min: np.ndarray
    node_max: np.ndarray
    node_left: np.ndarray
    node_right: np.ndarray
    node_start: np.ndarray
    node_count: np.ndarray
    order: np.ndarray  # int32[T] triangle permutation (leaf order)

    @property
    def num_nodes(self) -> int:
        """Number of flattened BVH nodes."""
        return int(self.node_min.shape[0])


def build_bvh(tri_min: np.ndarray, tri_max: np.ndarray, leaf_size: int = 8) -> BVH:
    """Build a median-split BVH from per-triangle AABBs.

    Args:
        tri_min / tri_max: ``float[T, 3]`` triangle AABB corners.
        leaf_size: max triangles per leaf.
    """
    t_min = np.asarray(tri_min, dtype=np.float64)
    t_max = np.asarray(tri_max, dtype=np.float64)
    centroid = 0.5 * (t_min + t_max)
    n_tri = t_min.shape[0]

    n_min, n_max = [], []
    n_left, n_right, n_start, n_count = [], [], [], []
    order: List[int] = []

    # Iterative build to avoid Python recursion limits on large scenes.
    # Each task: (parent_node_index, child_slot, triangle_index_array)
    # child_slot: 0 == root, 1 == left child of parent, 2 == right child.
    root_idx = [None]
    stack = [(None, 0, np.arange(n_tri, dtype=np.int64))]
    while stack:
        parent, slot, idxs = stack.pop()
        ni = len(n_min)
        bmin = t_min[idxs].min(axis=0)
        bmax = t_max[idxs].max(axis=0)
        n_min.append(bmin)
        n_max.append(bmax)
        n_left.append(-1)
        n_right.append(-1)
        n_start.append(-1)
        n_count.append(0)

        if parent is None:
            root_idx[0] = ni
        elif slot == 1:
            n_left[parent] = ni
        else:
            n_right[parent] = ni

        if idxs.shape[0] <= leaf_size:
            n_start[ni] = len(order)
            n_count[ni] = int(idxs.shape[0])
            order.extend(int(i) for i in idxs)
            continue

        c = centroid[idxs]
        axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
        sorted_local = idxs[np.argsort(c[:, axis], kind="stable")]
        mid = sorted_local.shape[0] // 2
        # Push right first so left is processed next (order is cosmetic).
        stack.append((ni, 2, sorted_local[mid:]))
        stack.append((ni, 1, sorted_local[:mid]))

    return BVH(
        node_min=np.asarray(n_min, dtype=np.float32),
        node_max=np.asarray(n_max, dtype=np.float32),
        node_left=np.asarray(n_left, dtype=np.int32),
        node_right=np.asarray(n_right, dtype=np.int32),
        node_start=np.asarray(n_start, dtype=np.int32),
        node_count=np.asarray(n_count, dtype=np.int32),
        order=np.asarray(order, dtype=np.int32),
    )
