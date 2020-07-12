# https://medium.com/analytics-vidhya/shap-part-3-tree-shap-3af9bcd7cd9b#:~:text=SHAP%20(SHapley%20Additive%20exPlanation)%20is,from%20it's%20individual%20feature%20values.&text=f%E2%82%9B()%20represents%20the%20prediction,model%20for%20the%20subset%20S.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree

import tree_shap


def toy_tree():
    data = pd.DataFrame(
        [
            [206, 216, 388, 10],
            [206, 246, 314, 10],
            [206, 127, 243, 10],
            [206, 331, 287, 10],
            [206, 168, 346, 10],
            [194, 272, 368, 20],
            [6, 299, 298, 50],
            [6, 299, 339, 50],
            [6, 301, 301, 30],
            [6, 301, 116, 30],
        ],
        columns=[0, 1, 2, "y"],
    )

    X_train = data[[0, 1, 2]]
    y_train = data["y"]

    tree = DecisionTreeRegressor(
        criterion="mae",
        max_depth=2,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=100,
    )

    tree.fit(X=X_train, y=y_train)
    return tree.tree_


# assert tree_shap.naive_tree_shap(tree.tree_, current_node=0, idx=0, val=150)[0][0] == 20
# assert tree_shap.naive_tree_shap(tree.tree_, current_node=0, idx=1, val=75)[0][0] == 27
