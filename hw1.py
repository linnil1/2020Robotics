# HW1 for 108-2 robotic
import numpy as np

# generate two random vector with size=3
np.random.seed(1)
a = np.random.randint(-10, 10, 3)
b = np.random.randint(-10, 10, 3)
print("Two random vector")
print(a, b)

print("pairwise multiply")
print(a * b)

print("dot")
print(np.dot(a, b))

print("cross product")
print(np.cross(a, b))

print("transpose")
print(a.T, b.T)

print("outer product")
s = np.outer(a, b)
print(s)

# eigen decomposition
print("eigen decomposition")
e_value, e_vector = np.linalg.eig(s)
print(e_vector)
print("And it's correspoding value")
print(e_value)
