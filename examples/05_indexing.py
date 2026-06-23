"""Indexing and slicing examples."""

import numpy as np
from apl_pruning import MiniAPLParser

W = np.array([[1.0, -2.0, 3.0, 4.0],
              [-0.5, 4.0, -1.0, 2.0],
              [3.0, 1.0, -2.0, 0.5]], dtype=np.float32)

parser = MiniAPLParser()
parser.set_variables(W=W)

# Row access
row0 = parser.evaluate("W[0]")
print(f"W[0] = {row0}")

# Column access
col1 = parser.evaluate("W[:, 1]")
print(f"W[:, 1] = {col1}")

# Element access
elem = parser.evaluate("W[1, 2]")
print(f"W[1, 2] = {elem}")

# Slice
slice_result = parser.evaluate("W[0:2]")
print(f"W[0:2] = {slice_result}")
