import numpy as np
import pandas as pd

print("=" * 50)
print("NUMPY FUNDAMENTALS")
print("=" * 50)

# ----------------------------
# 1D Array
# ----------------------------
arr1 = np.array([1, 2, 3, 4, 5])

print("\n1D Array")
print(arr1)
print("Shape:", arr1.shape)

# ----------------------------
# 2D Array
# ----------------------------
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\n2D Array")
print(arr2)
print("Shape:", arr2.shape)

# ----------------------------
# 3D Array
# ----------------------------
arr3 = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

print("\n3D Array")
print(arr3)
print("Shape:", arr3.shape)

# ----------------------------
# Broadcasting
# ----------------------------
print("\nBroadcasting")

broadcast = arr1 + 5
print(broadcast)

# ----------------------------
# Vectorized Operations
# ----------------------------
print("\nVectorized Operations")

print("Addition")
print(arr1 + 10)

print("Multiplication")
print(arr1 * 2)

print("Square")
print(arr1 ** 2)

# ----------------------------
# Matrix Multiplication
# ----------------------------
print("\nMatrix Multiplication")

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

result = np.matmul(A, B)

print(result)

# ----------------------------
# Read CSV Dataset
# ----------------------------
print("\nReading CSV Dataset")

df = pd.read_csv("sample_data.csv")

print(df)

# ----------------------------
# Mean
# ----------------------------
print("\nMean")

print(df.mean(numeric_only=True))

# ----------------------------
# Standard Deviation
# ----------------------------
print("\nStandard Deviation")

print(df.std(numeric_only=True))

# ----------------------------
# Correlation
# ----------------------------
print("\nCorrelation")

print(df.corr(numeric_only=True))

print("\nProgram Executed Successfully")