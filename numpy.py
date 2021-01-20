import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
(x_train , y_train), (x_test, y_test) = mnist.load_data()


def initialize_parameters(layer_dims):
    parameters = {}
    np.random.seed(3)
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ## w = [layer, layer-1]
        # b = [layer, 1]

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A,W,b):
    Z= np.dot(W,A)+b
    assert(Z.shape ==(W.shape[0],A.shape[1]))
    cache = {}
    cache["A"] = A
    # A_prev = [layer-1], m
    # Z = layer,m
    return Z, cache


def relu(Z):
    A= np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = {}
    cache["Z"] = Z
    return A, cache


def linear(Z):
    A = Z
    cache = {}
    cache["Z"] = Z
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == 'linear':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = linear(Z)
        # A = (layer, m)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = {}
    cache["linear_cache"] = linear_cache
    cache["activation_cache"] = activation_cache
    return A, cache


def l_model_forward(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)
    Al, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="linear")
    caches.append(cache)
    assert (Al.shape == (1, X.shape[1]))

    return Al, caches


def softmax_cross_entropy_loss(Z, Y=np.array([])):

    A = np.exp(Z - np.max(Z,axis = 0)) / np.sum(np.exp(Z-np.max(Z,axis = 0)),axis = 0,keepdims = True)
    # print "A : ",A
    if Y.shape[0] == 0:
        loss = []
    else:
        loss = -np.sum(Y*np.log(A+1e-8))   / A.shape[1]
    # loss = 0.05
    cache = {}
    cache["A"] = A
    return A, cache, loss


def softmax_cross_entropy_loss_der(Y, cache):
    A = cache["A"]
    dZ = A - Y
    return dZ


def linear_backward(dZ, cache, W, b):

    A_prev = cache["A"]
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)



    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_der(dA, cache):
    dZ = np.array(dA, copy=True)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def linear_activation_backward(dA, cache, W, b, activation):
    linear_cache = cache["linear_cache"]
    activation_cache = cache["activation_cache"]
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, W, b)
    elif activation == "linear":
        dZ = linear_der(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, W, b)

    return dA_prev, dW, db


def L_model_backwards(dAl, caches, parameters):
    grads = {}
    L = len(caches)
    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAl, current_cache,parameters["W"+str(L)],parameters["b"+str(L)] , activation = "linear")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache,parameters["W"+str(l+1)],parameters["b"+str(l+1)] ,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate) * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate) * grads["db" + str(l + 1)]

    return parameters


def one_hot(Y,num_classes):
    Y_one_hot = np.zeros((num_classes,Y.shape[1]))
    for i in range(Y.shape[1]):
        Y_one_hot[int(Y[0,i]),i] = 1
    return Y_one_hot


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009

    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layers_dims)


    num_classes = 10
    Y_one_hot = one_hot(Y, num_classes)


    for i in range(0, num_iterations):


        Al, caches = l_model_forward(X, parameters)

        A, cache_cross, cost = softmax_cross_entropy_loss(Z, Y_one_hot)
        dZ = softmax_cross_entropy_loss_der(Y_one_hot, cache_cross)

        grads = L_model_backwards(dZ, caches, parameters)

        parameters = update_parameters(parameters, grads, learning_rate)


        if i % 10 == 0:
            costs.append(cost)
        if i % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))

    return costs, parameters
    return parameters


def main():


    layer_dims = [784, 20, 7, 5, 1]


    learning_rate = 0.2
    num_iterations = 500

    costs, parameters = L_layer_model(x_train, y_train, layer_dims, learning_rate=learning_rate,
                                      num_iterations=num_iterations)

    X = range(0, 500, 10)
    plt.plot(X, costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()