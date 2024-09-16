from audioop import error

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerPerceptron:
    def __init__(self, n, m):
        # Two-layer perceptron with n inputs and m hidden units
        # Bias is the last element of w and v

        self.w = np.random.randn(n + 1, m)  # Adding bias to the input weights
        self.v = np.random.randn(m + 1)  # Adding bias to the output layer

    def forward(self, x, visualize=True):
        # Add bias to the input
        x = np.append(x, 1.0)
        # Compute the hidden layer output with tanh activation
        hidden_output = np.tanh(np.dot(self.w.T, x))
        # Add bias to the hidden layer output
        hidden_output = np.append(hidden_output, 1.0)

        # Compute the final output
        return np.dot(self.v, hidden_output)

    def single_train(self, x, y, lr):
        # Forward pass
        x = np.append(x, 1.0)  # Append bias

        hidden_output = np.tanh(np.dot(self.w.T, x))  # Hidden layer output

        hidden_output = np.append(hidden_output, 1.0)  # Append bias to hidden output

        output = np.dot(self.v, hidden_output)  # Final output

        # Compute the error
        error = y - output

        # Compute the gradient for the output layer
        delta_v = lr * error * hidden_output

        # Compute the gradient for the hidden layer
        hidden_without_bias = hidden_output[:-1]  # Exclude bias term
        delta_w = lr * np.outer(x, (1 - hidden_without_bias ** 2) * (self.v[:-1] * error))

        # Return updated weights
        return delta_w, delta_v, error ** 2


    def train(self, X, y, lr=1e-3, epochs=10, visualize=False, batch=True):
        loss = []
        # Train the network
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print(f"Epoch: {epoch + 1}/{epochs}", end="\r")
            error = 0
            if batch:
                delta_w = np.zeros_like(self.w)
                delta_v = np.zeros_like(self.v)
                for i in range(len(X)):
                    dw, dv, err = self.single_train(X[i], y[i], lr)
                    error += err
                    delta_w += dw
                    delta_v += dv
                self.w += delta_w
                self.v += delta_v
            else:
                for i in range(len(X)):
                    dw, dv, err = self.single_train(X[i], y[i], lr)
                    error += err
                    self.w += dw
                    self.v += dv
            loss.append(error / len(X))


            # Compute the loss
            if visualize:
                self.visualize(X, y)

        print("Last loss: ", loss[-1])
        return loss

    def vis_error(self, error):
        plt.plot(error)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.show()


    def visualize(self, X_in=None, y_in=None):
        # Visualize the output as color from red to blue in 2D space if n=2
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = self.forward([x[i], y[j]])

        # center of the color map is 0
        # set background color to white
        plt.gca().set_facecolor('white')

        plt.contourf(X, Y, Z, cmap="bwr", alpha=0.5, levels=[-5, -2, -1, 0, 1, 2, 5])
        plt.colorbar()
        if X_in is not None and y_in is not None:
            plt.scatter(X_in[:, 0], X_in[:, 1], c=y_in, cmap='bwr')
        plt.show()

    def test(self, X, y):
        # Test the network
        error = 0
        missed = 0
        for i in range(len(X)):
            output = self.forward(X[i])
            if np.sign(output) != np.sign(y[i]):
                missed += 1
            error += (y[i] - output) ** 2
        return error / len(X), missed / len(X)


if __name__ == "__main__":
    # generate some X and y
    X = np.random.randn(100, 2)

    # Generate y as a function of X
    y = np.zeros(100)
    for i in range(100):
        if X[i, 0] + X[i, 1] > 0:
            y[i] = 1
        else:
            y[i] = -1



    mlp = TwoLayerPerceptron(2, 12)

    loss = mlp.train(X, y, lr=1e-4, epochs=4000, visualize=False, batch=False)

    mlp.visualize(X, y)

