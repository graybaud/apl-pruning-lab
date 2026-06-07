"""Export APL formulas to PyTorch code."""

from apl_pruning import MiniAPLParser, to_pytorch, to_pytorch_function

parser = MiniAPLParser()

# Simple expression
code = "|W| x mean(|act|)"
print("APL:", code)
print("PyTorch:", to_pytorch(code))
print()

# With axis
code = "mean(|W|, dim=-1)"
print("APL:", code)
print("PyTorch:", to_pytorch(code))
print()

# Complete function
print("=== Complete PyTorch function ===")
print(to_pytorch_function("|W| x mean(|act|)", "wanda"))
print()

# Three-component method
code = """
direction <- (max(|W|)) / mean(|W|)
selectivity <- var(act) / mean(act)
distortion <- norm(S_full - S_without) / norm(S_full)
direction x selectivity x distortion
"""
print("APL multi-line:")
print(code)
print("PyTorch (first line):", to_pytorch("direction <- (max(|W|)) / mean(|W|)"))
print()

# Using the parser method directly
print("=== Using parser.to_pytorch() ===")
print("Wanda:", parser.to_pytorch("|W| x mean(|act|)"))
print("Softmax:", parser.to_pytorch("softmax(W, dim=-1)"))
print("Threshold:", parser.to_pytorch("threshold(|W|, 0.5)"))
print("TopK:", parser.to_pytorch("topk(|W|, 10)"))
print("Index:", parser.to_pytorch("W[0, :]"))
