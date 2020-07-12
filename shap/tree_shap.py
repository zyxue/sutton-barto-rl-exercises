def naive_tree_shap(tree, current_node, idx, val):
    """
    features: an ordered dictionary of index => feature value
    """
    current_feature = tree.feature[current_node]
    if current_feature == -2:  # a leaf node
        return tree.value[current_node]

    if idx == current_feature:
        if val <= tree.threshold[current_node]:
            current_node = tree.children_left[current_node]
        else:
            current_node = tree.children_right[current_node]
        return naive_tree_shap(tree, current_node, idx, val)
    else:
        current_node_samples = tree.n_node_samples[current_node]

        left_node = tree.children_left[current_node]
        right_node = tree.children_right[current_node]

        left_weight = tree.n_node_samples[left_node] / current_node_samples
        right_weight = tree.n_node_samples[right_node] / current_node_samples

        left = left_weight * naive_tree_shap(tree, left_node, idx, val)
        right = right_weight * naive_tree_shap(tree, right_node, idx, val)
        return left + right
