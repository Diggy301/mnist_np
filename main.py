import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from draw import drawingWindow as dW

class nn():
    def __init__(self):

        np.random.seed(1301)
        self.l1 = self.layer_init(784, 128)
        self.l2 = self.layer_init(128, 10)

        self.learning_rate = 0.001
        self.bias = 128
        self.losses, self.accuracies = [], []

    def layer_init(self, m, h):
        # initialized like torch linear layer 
        layer = np.random.uniform(-1., 1., size=(m,h))/np.sqrt((m*h))

        return layer.astype(np.float32)

    def forward_backward(self, x, y):
        out = np.zeros((len(y),10), np.float32) 
        out[range(out.shape[0]),y] = 1


        # forward pass
        x_l1 = x.dot(self.l1)
        x_relu = np.maximum(x_l1, 0)
        x_l2 = x_relu.dot(self.l2)
        
        # logsoftmax as activation function
        x_lsm = x_l2 - self.logsumexp(x_l2).reshape((-1,1))
        x_loss = (-out * x_lsm).mean(axis=1)


        # backward pass
        d_out = -out / len(y)

        # derivate of logsoftmax
        dx_lsm = d_out - np.exp(x_lsm)*d_out.sum(axis=1).reshape((-1,1))

        # derivate of l2
        d_l2 = x_relu.T.dot(dx_lsm)
        # derivate of relu
        dx_relu = dx_lsm.dot(self.l2.T)
        # derivate of l1
        dx_l1 = (x_relu > 0).astype(np.float32) * dx_relu
        d_l1 = x.T.dot(dx_l1)

        return x_loss, x_l2, d_l1, d_l2

    def logsumexp(self, x):
        # http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

        c = x.max(axis=1)
        return c + np.log(np.exp(x-c.reshape((-1,1))).sum(axis=1))


    def forward(self, x):
        x = x.dot(self.l1)
        x = np.maximum(x, 0)
        x = x.dot(self.l2)  
        return x

    def numpy_eval(self, test_data, test_labels):
        Y_test_preds_out = self.forward(test_data.reshape((-1, 28*28)))
        Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
        return (test_labels == Y_test_preds).mean()

    def train(self, x_train, x_labels):
        for i in range(1000):
            sample = np.random.randint(0, x_train.shape[0], size=(self.bias))
            x = x_train[sample].reshape((-1, 28*28))
            y = x_labels[sample]
            x_loss, x_l2, d_l1, d_l2 = self.forward_backward(x, y)

            cat =np.argmax(x_l2, axis=1)
            accuracy = (cat == y).mean()


            # SGD - stochastic gradient descent
            self.l1 = self.l1 - self.learning_rate*d_l1
            self.l2 = self.l2 - self.learning_rate*d_l2

            loss = x_loss.mean()
            self.losses.append(loss)
            self.accuracies.append(accuracy)

    def plots(self):
        #plt.ylim([-0.1, 1.1])
        plt.plot(self.losses)
        plt.plot(self.accuracies)
        plt.show()




x_train     = idx2numpy.convert_from_file('samples/train-images.idx3-ubyte')
x_labels    = idx2numpy.convert_from_file('samples/train-labels.idx1-ubyte')
test        = idx2numpy.convert_from_file('samples/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('samples/t10k-labels.idx1-ubyte')

NN = nn()
NN.train(x_train, x_labels)



w = dW(28, 20)
mynumber = w.mainloop().T



x = mynumber.reshape(1,-1).dot(NN.l1)
x = np.maximum(x, 0)
x = x.dot(NN.l2)
print(np.argmax(x))