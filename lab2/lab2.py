from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(100)

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)



def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels. append(0)
        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)



def show_result(x, y, pred_y):

    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


def loss_func(y, y_hat):
    return (y - y_hat) * (y - y_hat)

def derivtive_loss(y, y_hat):
    return -2*(y - y_hat)/y.shape[0]

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)


x1, y1 = generate_linear()
x2, y2 = generate_XOR_easy()
#[100 2] [100 1] [21 2] [21 1]

#parameter
n_epochs = 200
lr = 1

#weight in three layers
w1 = np.random.normal(size=(2,6))
w2 = np.random.normal(size=(6,4))
w3 = np.random.normal(size=(4,1))

w1_grad = np.zeros((2,6))
w2_grad = np.zeros((6,4))
w3_grad = np.zeros((4,1))
#######################################
# training of linear data [2  6   4  1]
#-------------------------[ 26  64  41 ]
loss_arr1 = []
for epoch in range(n_epochs):
    total_loss = 0
    for i in range(100):
        #moving forward 
        a1 = sigmoid(np.matmul(x1[i], w1)).reshape((1,-1))
        a2 = sigmoid(np.matmul(a1, w2))
        a3 = sigmoid(np.matmul(a2, w3))

        #count loss
        loss = loss_func(a3.reshape((1)), y1[i])
        total_loss += loss
        print("epoch: ", epoch, "loss: ",total_loss/(i+1))
        loss_arr1.append(total_loss/(i+1))
        #count gradient
        w3_grad = a2.T @ derivative_sigmoid(a3)*derivtive_loss(a3, y1[i])
        w2_grad = a1.T @ (derivative_sigmoid(a2)*((derivative_sigmoid(a3)*derivtive_loss(a3, y1[i])) @ w3.T))   
        w1_grad = x1[i].reshape((1,-1)).T @ (derivative_sigmoid(a1) * ((derivative_sigmoid(a2) * ((derivative_sigmoid(a3) * derivtive_loss(a3, y1[i])) @ w3.T)) @ w2.T))

        #weight update
        w1 += w1_grad
        w2 += w2_grad
        w3 += w3_grad

#testing of linear data
pred_y1 = np.zeros((100,1))
n_correct1 = 0
for i in range(100):
    p1 = sigmoid(np.matmul(x1[i], w1)).reshape((1,-1))
    p2 = sigmoid(np.matmul(p1, w2))
    p3 = sigmoid(np.matmul(p2, w3))
    print(p3)
    pred_y1[i] = p3
    if abs(p3-y1[i])<0.5:
        n_correct1 += 1


#weight clear
w1 = np.random.normal(size=(2,6))
w2 = np.random.normal(size=(6,4))
w3 = np.random.normal(size=(4,1))

w1_grad = np.zeros((2,6))
w2_grad = np.zeros((6,4))
w3_grad = np.zeros((4,1))

###################################
#training of xor data[2  6   4   1]
#--------------------[ 26  64  41 ]
loss_arr2 = []
for epoch in range(n_epochs):
    total_loss = 0
    for i in range(21):
        #moving forward 
        a1 = sigmoid(np.matmul(x2[i], w1)).reshape((1,-1))
        a2 = sigmoid(np.matmul(a1, w2))
        a3 = sigmoid(np.matmul(a2, w3))

        #count loss
        loss = loss_func(a3.reshape((1)), y2[i])
        total_loss += loss
        print("epoch: ", epoch, "loss: ",total_loss/(i+1))
        loss_arr2.append(total_loss/(i+1))
        #count gradient
        w3_grad = a2.T @ derivative_sigmoid(a3)*derivtive_loss(a3, y2[i])
        w2_grad = a1.T @ (derivative_sigmoid(a2)*((derivative_sigmoid(a3)*derivtive_loss(a3, y2[i])) @ w3.T))   
        w1_grad = x2[i].reshape((1,-1)).T @ (derivative_sigmoid(a1) * ((derivative_sigmoid(a2) * ((derivative_sigmoid(a3) * derivtive_loss(a3, y2[i])) @ w3.T)) @ w2.T))

        #weight update
        w1 += w1_grad
        w2 += w2_grad
        w3 += w3_grad

#testing of linear data
pred_y2 = np.zeros((100,1))
n_correct2 = 0
for i in range(21):
    p1 = sigmoid(np.matmul(x2[i], w1)).reshape((1,-1))
    p2 = sigmoid(np.matmul(p1, w2))
    p3 = sigmoid(np.matmul(p2, w3))
    print(p3)
    pred_y2[i] = p3
    if abs(p3-y2[i])<0.5:
        n_correct2 += 1

        
plt.title('linear data loss', fontsize=18)
plt.plot(range(n_epochs*100),loss_arr1)
plt.show()

plt.title('XOR data loss', fontsize=18)
plt.plot(range(n_epochs*21),loss_arr2)
plt.show()

print("linear data accuracy: ", n_correct1/100, "XOR data accuracy: ", n_correct2/21)
show_result(x1, y1, pred_y1)
show_result(x2, y2, pred_y2)










    