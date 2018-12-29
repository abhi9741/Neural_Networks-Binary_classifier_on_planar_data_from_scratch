# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) 

X, Y = load_planar_dataset()

def visualise() :
	plt.scatter(X[0, :], X[1, :], c=Y.reshape(400,), s=40, cmap=plt.cm.Spectral)
	plt.show()
def dataset_dimensions():
    X.shape
    m=X.shape[1]
    n=X.shape[0]
    print("Dimensions of input matrix X = ",X.shape)
    print("Number of training examples = ",m)
    print("Dimensions of input vector = ",n," Dimensional")
    print("Dimensions of input matrix Y = ",Y.shape)
dataset_dimensions()
print("-"*100)
i=input("do you want to visualise the data [y/n] :")


if i=='y' :
	visualise()

def initialize_parameters(n_x, n_h, n_y):
        
    np.random.seed(2)     
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
           
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


def compute_cost(A2, Y, parameters):
    
    
    
    m = Y.shape[1] 
    logprobs = Y*np.log(A2)+(1-Y)*np.log(1-A2)
    cost = (-1/m)*np.sum(logprobs)
    
    
    cost = np.squeeze(cost)      
                                
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
   
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2-Y
    dW2 = (1/m)*(np.dot(dZ2,A1.T))
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1-learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    costs =[]

    print("-"*80)
    print("Neural network structure :")
    print("Number of hidden layers = ",1)
    print("Number of units in hidden layer = ",n_h)
    print("Number of units in output layer = ",n_y)
    print("-"*80)
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
   

    for i in range(0, num_iterations):
         
        
        A2, cache = forward_propagation(X,parameters)
        
        cost = compute_cost(A2,Y,parameters)
        costs.append(cost)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)
        
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    final_out = {"parameters": parameters,
                  "costs": costs,
                  }
    
    return final_out



def predict(parameters, X):
    
    A2, cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
   
    
    return predictions

final_out = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
parameters = final_out["parameters"]
costs = final_out["costs"]

i=input("do you want to visualise the gradient descent at current learning rate [y/n] :")
if i=='y' :
	costs = np.squeeze(costs)
	plt.plot(costs)
	plt.xlabel("Cost")
	plt.ylabel("number of iterations")
	plt.title("Gradient descent at learning rate = 1.2")
	plt.show()

i=input("do you want to visualise the model learnt [y/n] :")
if i=='y' :

	plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.reshape(400,))
	plt.title("Decision Boundary for hidden layer size " + str(4))
	plt.show()

print("-"*80)
predictions = predict(parameters, X)
print ('Accuracy of the model: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# This may take about 2 minutes to run





i=input("do you want to check the neural network performance for different number of units [y/n] :")
if i=='y' :
	plt.figure(figsize=(16, 32))
	hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
	# hidden_layer_sizes=input("give the list of number of hidden units :")
	for i, n_h in enumerate(hidden_layer_sizes):
	    plt.subplot(5, 2, i+1)
	    plt.title('Hidden Layer of size %d' % n_h)
	    final_out = nn_model(X, Y, n_h, num_iterations = 5000)
	    parameters = final_out["parameters"]
	    costs = final_out["costs"]
	    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.reshape(400,))
	    predictions = predict(parameters, X)
	    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
	    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
	plt.show()