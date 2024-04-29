from typing import List, NamedTuple, Set

import numpy as np
from numpy.typing import ArrayLike
from sklearn.tree._tree import Tree


def naive_tree_shap(tree, current_node, features):
    """
    features: an dictionary of feature index => feature value
    """
    current_feature_index = tree.feature[current_node]
    if current_feature_index == -2:  # a leaf node
        return tree.value[current_node][0][0]

    if current_feature_index in features:
        current_feature_val = features[current_feature_index]
        if current_feature_val <= tree.threshold[current_node]:
            current_node = tree.children_left[current_node]
        else:
            current_node = tree.children_right[current_node]
        return naive_tree_shap(tree, current_node, features)
    else:
        current_node_samples = tree.n_node_samples[current_node]

        left_node = tree.children_left[current_node]
        right_node = tree.children_right[current_node]

        left_weight = tree.n_node_samples[left_node] / current_node_samples
        right_weight = tree.n_node_samples[right_node] / current_node_samples

        left = left_weight * naive_tree_shap(tree, left_node, features)
        right = right_weight * naive_tree_shap(tree, right_node, features)
        return left + right


def exp_value(x: ArrayLike, S: Set[int], tree: Tree) -> float:
    """Implements Estimation of E[f(x)|x_S].

    The algorithm is described in Methods 10.1.1, Algorithm 1 in
    https://arxiv.org/pdf/1905.04610.pdf. The notations here follows closely to
    those described in the paper.

    Args:
        x: feature vector of an instance to be explained.
        S: set of indices that specify the features to use in x.
        tree: a trained decision tree.

    Returns:
        the tree prediction for the given feature set.
    """

    # Interpreted shape according to the docstring of tree.value.
    node_count, n_outputs, max_n_classes = tree.value.shape
    assert n_outputs == 1, f"expected n_outputs to be 1, found {n_outputs}"
    assert max_n_classes == 1, f"expected max_n_classes to be 1, found {max_n_classes}"

    # All vectors are of length of the number of nodes in the tree. In sklearn,
    # nodes are index in a depth-first fashion.

    # The calculation of value for each node depends on criterio, see implementation
    # of node_value methods in
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx
    # To calculate SHAP, only values of leaf nodes are used.
    vec_v = tree.value.ravel()
    vec_a = tree.children_left
    vec_b = tree.children_right
    vec_t = tree.threshold  # threshold used in each node.
    vec_d = tree.feature  # index of the feature used in each node
    vec_r = tree.n_node_samples  # aka. cover: # data samples that fall in the subtree

    def G(j: int) -> float:
        """
        Args:
            j: the node index in the tree.
        """
        if vec_a[j] == -1 and vec_b[j] == -1:
            # these mean the same thing as the if condition, i.e. it's a leaf node.
            assert vec_t[j] == -2
            assert vec_d[j] == -2
            out = vec_v[j]
        else:
            aj = vec_a[j]
            bj = vec_b[j]
            if vec_d[j] in S:
                if x[vec_d[j]] <= vec_t[j]:
                    out = G(aj)
                else:
                    out = G(bj)
            else:
                assert vec_r[j] == vec_r[aj] + vec_r[bj]
                out = (G(aj) * vec_r[aj] + G(bj) * vec_r[bj]) / vec_r[j]

        return out

    return G(0)


class Feature(NamedTuple):

    d: int  # feature index
    z: float  # fraction of zero paths
    o: float  # fraction of one paths
    w: float  # proportion of sets of a given cardinality that are present, weighted by their Shapley weight.


# def exp_value(x: ArrayLike, tree: Tree) -> List[float]:
#     """Implementes the Tree SHAP as described in 10.1.2 Algorithm2."""
#     # vector of SHAP values for each feature
#     phis = np.zeros_like(x)

#     vec_v = tree.value.ravel()
#     vec_a = tree.children_left
#     vec_b = tree.children_right
#     vec_t = tree.threshold  # threshold used in each node.
#     vec_d = tree.feature  # index of the feature used in each node
#     vec_r = tree.n_node_samples  # aka. cover: # data samples that fall in the subtree

#     def recurse(j: int, m: List[Feature], pz: float, po: float, pi: int) -> float:
#         """
#         Args:
#             j: index of node
#             m: path of unique features we have split on so far
#             pz: fraction of zeros that are going to extend the subsets.
#             po: fraction of ones that are going to textend the subsets
#             pi: index of the feature used to make the last split.
#         """
#         # TODO
#         pass

#     def extend(m: List[Feature], pz: float, po: float, pi: int) -> List[Feature]:
#         pass

#     def unwind(m: List[Feature], i: int) -> List[Feature]:
#         pass
