import numpy as np
x = [1,2]
y = [5,0]

def euclidean(a, b):
    diff = np.array(a) - np.array(b) 
    square = diff ** 2 
    summation = sum(square)
    return np.sqrt(summation)

print(x)
print(y)
print(euclidean(x,y))
