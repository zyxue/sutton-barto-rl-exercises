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
