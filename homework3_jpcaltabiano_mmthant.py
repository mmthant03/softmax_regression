import numpy as np
import matplotlib.pyplot as plt

DEBUG = False
image_path = "./image/"

# Cross Entropy Loss
def fCE(yhat, y):
    return -np.mean(np.log(yhat[y==1]))
# Percent Correct
def fPC(yhat, y):
    return np.mean(y.argmax(axis=1) == yhat.argmax(axis=1))

def predictor(w, X):
    z = np.exp(X.dot(w))
    z_sum = np.sum(z, axis=1).reshape(-1, 1)
    yhat = z/z_sum
    return yhat

def SGD(w, X, y):
    yhat = predictor(w, X)
    n = y.shape[0]
    w = X.T.dot(yhat-y)/n
    return w

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    n = trainingLabels.shape[0]
    W = 0.01 * np.random.randn(785,10)
    for n_tilde in range(n // batchSize):
        mini_x = trainingImages[n_tilde*batchSize: (n_tilde+1)*batchSize]
        mini_y = trainingLabels[n_tilde*batchSize: (n_tilde+1)*batchSize] 
        W = W - epsilon*SGD(W, mini_x, mini_y)
    return W

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...
    trainingImages = np.insert(trainingImages, 784, 1, axis=1)
    testingImages = np.insert(testingImages, 784, 1, axis=1)

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)

    train_ce = fCE(predictor(W, trainingImages), trainingLabels)
    test_ce = fCE(predictor(W, testingImages), testingLabels)
    train_pc = fPC(predictor(W, trainingImages), trainingLabels)
    test_pc = fPC(predictor(W, testingImages), testingLabels)

    print('training loss:\t {},\t training accuracy:\t {}'.format(train_ce, train_pc))
    print('testing  loss:\t {},\t testing accuracy:\t {}'.format(test_ce, test_pc))
    
    # Visualize the vectors
    # ...
    i = 0
    for w in W[:784].T:
        wi = w.reshape((28, 28))
        image_file = "w"+str(i)+".png"
        plt.imshow(wi)
        plt.savefig(image_path+image_file)
        i = i+1