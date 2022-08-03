import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


def generate_linear(n=100):
    np.random.seed(0)
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    np.random.seed(0)
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, y_pred):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if y_pred[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(y):
    return np.multiply(y, 1 - y)


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return x > 0


class Layer:
    def __init__(self, num_input, num_output, with_bias, activation) -> None:
        self.weight = np.random.randn(num_input, num_output)
        self.bias = np.atleast_2d(np.random.randn(num_output))
        self.with_bias = with_bias
        self.activation = activation
        self.XW = None
        self.XWa = None
        self.da = None
        self.recur = None

    def compute_XWa(self, X):
        self.XW = np.matmul(X, self.weight)
        if self.XW.shape != self.bias.shape:
            multiplier = self.XW.shape[0]
            self.bias = np.repeat(self.bias, multiplier, axis=0)

        if self.with_bias == 1:
            self.XW += self.bias
        if self.activation == "sigmoid":
            self.XWa = sigmoid(self.XW)
        elif self.activation == "relu":
            self.XWa = relu(self.XW)
        else:
            self.XWa = self.XW
        return self.XWa

    def compute_da(self):
        if self.activation == "sigmoid":
            self.da = derivative_sigmoid(self.XWa)
        elif self.activation == "relu":
            self.da = derivative_relu(self.XWa)
        else:
            self.da = 1


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, num_input, num_output, with_bias, activation="sigmoid"):
        self.layers.append(Layer(num_input, num_output, with_bias, activation))

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.compute_XWa(X)  # compute XWa
            layer.compute_da()  # compute da

    def backprop(self, X, y, lr):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:  # output layer
                prev_layer = self.layers[i-1]
                layer.recur = layer.da * (y - layer.XWa)
                layer.delta = lr * np.matmul(prev_layer.XWa.T, layer.recur)

            elif layer == self.layers[0]:  # input layer
                next_layer = self.layers[i+1]
                layer.recur = layer.da * \
                    np.matmul(next_layer.recur, next_layer.weight.T)
                layer.delta = lr * np.matmul(X.T, layer.recur)

            else:  # hidden layers
                prev_layer = self.layers[i-1]
                next_layer = self.layers[i+1]
                layer.recur = layer.da * \
                    np.matmul(next_layer.recur, next_layer.weight.T)
                layer.delta = lr * np.matmul(prev_layer.XWa.T, layer.recur)
            layer.weight = np.add(layer.weight, layer.delta)

    def train(self, lr, type="XOR", num_epoch=20000):
        assert type == "linear" or type == "XOR"
        if type == "linear":
            X, Y = generate_linear()
        else:
            X, Y = generate_XOR_easy()
        mse_loss = []
        accuracy = []
        for i in range(num_epoch):
            self.feedforward(X)
            self.backprop(X, Y, lr)  # update all weights
            Y_prob = self.layers[-1].XWa
            mse = np.square(Y - Y_prob).mean()
            Y_pred = np.round(Y_prob)
            acc = 1 - np.sum(np.abs(Y - Y_pred))/len(Y)
            mse_loss.append(mse)
            accuracy.append(acc)

            if i % 500 == 0:
                print(
                    f'Epoch {i}, loss = {mse_loss[-1]}, accuracy = {accuracy[-1]:.2}')
            if acc == 1:
                print(i)
                return mse_loss, accuracy
        # show_result(X, Y, Y_pred)
        return mse_loss, accuracy

    def test(self, type="XOR"):
        assert type == "linear" or type == "XOR"
        if type == "linear":
            X, Y = generate_linear()
        else:
            X, Y = generate_XOR_easy()
        self.feedforward(X)
        Y_prob = self.layers[-1].XWa
        mse = np.square(Y - Y_prob).mean()
        Y_pred = np.round(Y_prob)
        acc = 1 - np.sum(np.abs(Y - Y_pred))/len(Y)
        print(f"Testing loss = {mse}, accuracy = {acc}")

        show_result(X, Y, Y_pred)


def plot_loss_curve(mse_loss, accuracy):
    plt.figure()
    plt.plot(mse_loss, color="r")
    plt.xlabel("Epoch")
    plt.legend(labels=["Linear training loss",
               "Linear training accuracy"], loc="best")
    plt.title("Linear Learning Curve")
    plt.show()


if __name__ == "__main__":
    # settings
    num_epoch = 15001
    activation = "sigmoid"
    with_bias = 1
    lr = 0.01
    neurons = 20
    index = 0
    if index == 1:
        type = "linear"
    else:
        type = "XOR"

    nn = NeuralNetwork()
    nn.add_layer(2, neurons, with_bias=with_bias, activation="sigmoid")
    nn.add_layer(neurons, neurons, with_bias=with_bias, activation=activation)
    nn.add_layer(neurons, neurons, with_bias=with_bias, activation=activation)
    nn.add_layer(neurons, 1, with_bias=with_bias, activation="sigmoid")

    mse_loss, accuracy = nn.train(lr=lr, type=type, num_epoch=num_epoch)
    plot_loss_curve(mse_loss, accuracy)
    nn.test(type=type)
