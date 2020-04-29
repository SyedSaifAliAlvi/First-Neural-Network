import numpy as np

def sigmoid(x,deriv=False):
  if (deriv ==True):
    return x*(1-x)
  return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],
              [1],
              [0],
              [0]])

X_shape=np.shape(X)
y_shape=np.shape(y)
print(X_shape) #(4,3)
print(y_shape) #(4,1)

w0 = 2*np.random.random((3,4)) -1
w1 = 2*np.random.random((4,1)) -1

print(w0)
print(w1)

l0 = X
l1 = sigmoid(np.dot(l0,w0))
l2 = sigmoid((np.dot(l1,w1)))



print('l0 = ',l0)
print('l1 = ',l1)
print('l2 = ',l2)
print()
print('l0 shape = ',np.shape(l0))
print('l1 shape = ',np.shape(l1))
print('l2 shape = ',np.shape(l2))
print()
l2_error = y-l2
print('l2_error = ',l2_error)
print()
print('l2_error shape = ',np.shape(l2_error))

l2_delta = l2_error*sigmoid(l2,deriv=True) #Multiplying the slope from sigmoid by the error.

print('l2_delta = ',l2_delta)
print()
print('l2_delta shape = ',np.shape(l2_delta))
print()
l1_error = l2_delta.dot(w1.T)

print('l1_error = ',l1_error)
print()
print('l1_error shape = ',np.shape(l1_error))
print()

l1_delta = l1_error*sigmoid(l1,deriv=True)
print('l1_delta = ',l1_delta)
print()
print('l1_delta shape = ',np.shape(l1_delta))
print()

print('l1.T.dot(l2_delta) = ',l1.T.dot(l2_delta))
print()
print('l1.T.dot(l2_delta) SHAPE = ',np.shape(l1.T.dot(l2_delta)))
print()
print('l0.T.dot(l1_delta) = ',l0.T.dot(l1_delta))
print('l0.T.dot(l1_delta) SHAPE = ',np.shape(l0.T.dot(l1_delta)))
print()
w1 = w1+l1.T.dot(l2_delta)
w0 = w0+l0.T.dot(l1_delta)

for i in range(70001):
  l0 = X
  l1 = sigmoid(np.dot(l0,w0))
  l2 = sigmoid((np.dot(l1,w1)))

  l2_error = y-l2
  if(i%10000==0):
    print('Cost after iteration',i,'=',str(np.mean(np.abs(l2_error))))

  l2_delta = l2_error*sigmoid(l2,deriv=True)

  l1_error = l2_delta.dot(w1.T)

  l1_delta = l1_error*sigmoid(l1,deriv=True)

  w1 = w1+l1.T.dot(l2_delta)
  w0 = w0+l0.T.dot(l1_delta)