get_ipython().run_cell_magic(
    magic_name="javascript",
    line="",
    cell='MathJax.Hub.Config({\n    TeX: { equationNumbers: { autoNumber: "AMS" } }\n});',
)

import altair as alt
import numpy as np
