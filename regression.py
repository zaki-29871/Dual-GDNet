import numpy as np

# ax
A = np.array([0, 1, 2]).reshape(-1, 1)
b = np.array([-1, -2, -3]).reshape(-1, 1)

x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
print(x)

# ax + b
A = np.array([[0, 1], [1, 1], [2, 1]]).reshape(-1, 2)
b = np.array([-1, -2, -3]).reshape(-1, 1)

x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
print(x)

# Correlation Coefficient
A = np.array([0, 1, 8])
B = np.array([0, 2, -1])
print(np.corrcoef(A, B)[0, 1])
