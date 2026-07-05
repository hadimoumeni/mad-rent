"""
Isolation Forest — implemented from scratch (no scikit-learn).

Faithful to Liu, Ting & Zhou, "Isolation Forest" (ICDM 2008) and the follow-up
"Isolation-Based Anomaly Detection" (TKDD 2012).

Core idea
---------
Anomalies are "few and different", so they are easier to isolate. If we build a
random binary tree by repeatedly picking a random feature and a random split
value, anomalous points get separated from the rest after only a few splits
(short path from the root), while normal points sit in dense regions and require
many splits (long path). Averaging the path length over many random trees yields
a robust anomaly score.

Nothing here uses labels — it is fully unsupervised.
"""

from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649015329


def _harmonic(i: float) -> float:
    """H(i) ≈ ln(i) + Euler–Mascheroni constant."""
    return np.log(i) + EULER_GAMMA


def c_factor(n: int) -> float:
    """
    Average path length of an unsuccessful search in a Binary Search Tree of n
    points. Used (a) to correct the path length at external nodes that still
    contain >1 point, and (b) to normalise the final score so it is independent
    of the sub-sample size. Eq. (1) in the paper.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * _harmonic(n - 1) - (2.0 * (n - 1) / n)


class _Node:
    """A node of one isolation tree.

    Internal nodes store the split (feature, threshold) and children.
    External (leaf) nodes store how many training points reached them, which
    feeds the c_factor path-length correction.
    """

    __slots__ = ("feature", "threshold", "left", "right", "size", "is_leaf", "depth")

    def __init__(self, depth: int):
        self.depth = depth
        self.is_leaf = False
        self.feature = -1
        self.threshold = 0.0
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.size = 0


class IsolationTree:
    """A single isolation tree (iTree) grown on a sub-sample."""

    def __init__(self, height_limit: int, rng: np.random.Generator):
        self.height_limit = height_limit
        self.rng = rng
        self.root: _Node | None = None

    def fit(self, X: np.ndarray) -> "IsolationTree":
        self.root = self._grow(X, current_depth=0)
        return self

    def _grow(self, X: np.ndarray, current_depth: int) -> _Node:
        node = _Node(current_depth)
        n = X.shape[0]

        # External node: stop when we hit the height limit or can't split further.
        if current_depth >= self.height_limit or n <= 1:
            node.is_leaf = True
            node.size = n
            return node

        # Pick a random split feature that actually has spread; if every feature
        # in this node is constant, we cannot separate the points -> leaf.
        n_features = X.shape[1]
        candidates = self.rng.permutation(n_features)
        chosen = None
        lo = hi = 0.0
        for q in candidates:
            lo = X[:, q].min()
            hi = X[:, q].max()
            if hi > lo:
                chosen = q
                break
        if chosen is None:
            node.is_leaf = True
            node.size = n
            return node

        # Random split value uniformly in (min, max) of the chosen feature.
        threshold = self.rng.uniform(lo, hi)
        left_mask = X[:, chosen] < threshold

        node.feature = int(chosen)
        node.threshold = float(threshold)
        node.left = self._grow(X[left_mask], current_depth + 1)
        node.right = self._grow(X[~left_mask], current_depth + 1)
        return node

    def path_length(self, x: np.ndarray) -> float:
        """Path length of a single point x (edges traversed + leaf correction)."""
        node = self.root
        while not node.is_leaf:
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        # Add the expected path through the sub-tree we never built out.
        return node.depth + c_factor(node.size)


class IsolationForest:
    """
    An ensemble of isolation trees.

    Parameters
    ----------
    n_trees : int
        Number of trees in the ensemble (t in the paper). More trees -> more
        stable scores; convergence is usually reached well before 200.
    sample_size : int
        Sub-sampling size (psi). The paper's key insight: 256 is enough, and
        small sub-samples actually *help* by reducing swamping/masking.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_trees: int = 200, sample_size: int = 256, random_state: int = 42):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.random_state = random_state
        self.trees: list[IsolationTree] = []
        self._c = 0.0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        psi = min(self.sample_size, n)
        # Height limit = average tree height for psi points; beyond this depth,
        # points are almost certainly normal so there is no value in growing.
        height_limit = int(np.ceil(np.log2(max(psi, 2))))
        self._c = c_factor(psi)

        master_rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_trees):
            idx = master_rng.choice(n, size=psi, replace=False)
            # Give each tree its own independent stream, seeded deterministically.
            tree_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            tree = IsolationTree(height_limit, tree_rng).fit(X[idx])
            self.trees.append(tree)
        return self

    def path_length_mean(self, X: np.ndarray) -> np.ndarray:
        """Mean path length E[h(x)] across all trees, per point."""
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            xi = X[i]
            total = 0.0
            for tree in self.trees:
                total += tree.path_length(xi)
            out[i] = total / len(self.trees)
        return out

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly score s(x) = 2 ** ( -E[h(x)] / c(psi) ), Eq. (2).

        s -> 1  : very short paths, strong anomaly.
        s ~ 0.5 : path length ~ average, no clear signal.
        s -> 0  : long paths, very normal.
        """
        eh = self.path_length_mean(X)
        return 2.0 ** (-eh / self._c) if self._c > 0 else np.full(X.shape[0], 0.5)


if __name__ == "__main__":
    # Smoke test: a dense Gaussian blob with a handful of planted outliers.
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(500, 2))
    outliers = np.array([[8, 8], [-7, 6], [6, -7], [0, 9]], dtype=float)
    X = np.vstack([normal, outliers])

    forest = IsolationForest(n_trees=200, sample_size=256, random_state=1).fit(X)
    scores = forest.anomaly_score(X)

    print("mean score (normal points):", round(scores[:500].mean(), 3))
    print("scores of planted outliers:", np.round(scores[500:], 3))
    ranked = np.argsort(-scores)[:6]
    print("top-6 most anomalous indices (planted are >=500):", ranked)
    assert set(range(500, 504)).issubset(set(ranked.tolist())), "outliers not surfaced!"
    print("OK: isolation forest surfaces planted outliers.")
