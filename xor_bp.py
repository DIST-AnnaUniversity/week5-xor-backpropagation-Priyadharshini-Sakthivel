import numpy as np
def sigmoid(n):
    result = 1 / (1 + np.exp(-n))
    return result

z = np.array([[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1],])
d = np.array([[0],[1],[1],[0],])
print("Weights V - should be ones")
v = np.ones((2,3))
print(v)
print("Weights W - should be zeros")
w = np.zeros((3,1))
print(w)
print(z)
print(d)
print(v)
print(w)
### Forward propagation
y_net = np.dot(v,z[0])
print(y_net)
y = sigmoid(y_net)
y = np.append(y,[1])
y = y.reshape(3,1)
wt = np.transpose(w)
print(y)
print(wt.shape)
print(y.shape)
out_net = np.dot(wt,y)
out = sigmoid(out_net)
print(out)

#error calculation
del_o = (d[0]-out)*(1-out)*out
print(del_o)
del_hid = del_o*(y)
print(del_hid)
