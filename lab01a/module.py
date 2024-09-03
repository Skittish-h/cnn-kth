import numpy as np
import matplotlib.pyplot as plt


class SingleLayerPerceptron:
    def __init__(self, n):
        # single layer perceptron with n inputs
        # bias is the last element of w
        self.w = np.random.randn(n + 1)  # Initialize with random values including negative ones

    def forward(self, x):
        # forward pass
        return np.dot(self.w, np.append(x, 1.0))

    def _update(self, x, y, lr, method="delta"): # note delta expects y to be -1 or 1 and perceptron expects 0 or 1
        dw = np.zeros_like(self.w)
        output = self.forward(x)
        # update the weights
        if method == "delta":
            error = y - output
            dw = lr * error * np.append(x, 1.0)
        elif method == "perceptron":
            if y == 1 and output < 0:
                dw = lr * np.append(x, 1.0)
            elif y == 0 and output >= 0:
                dw = -lr * np.append(x, 1.0)
        return dw

    def visualize(self, X, y):
        # visualize the data and the decision boundary
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

        # Plot the decision boundary
        x1_vals = np.array([-1, 2])
        x2_vals = -(self.w[0] * x1_vals + self.w[2]) / self.w[1]

        plt.plot(x1_vals, x2_vals, 'k--')
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)
        plt.show()

    def train(self, X, y, lr, method="delta", epochs=10, visualize=False):
        # Training loop
        for i in range(epochs):
            dw = np.zeros_like(self.w)
            for j in range(len(X)):
                dw += self._update(X[j], y[j], lr, method)
            print(f"Epoch {i}, Loss: {np.sum(dw)}")
            self.w += dw
            if visualize:
                self.visualize(X, y)

# simple test to check if the class works
if __name__ == "__main__":
    # X_positives = np.random.rand(100, 2) * 0.5 + 0.5
    # X_negatives = np.random.rand(100, 2) * 0.5
    # X = np.concatenate([X_positives, X_negatives])
    # y = np.concatenate([np.ones(100), np.zeros(100)])

    X_positives = np.random.rand(100, 2) * 0.5 + 0.25
    X_negatives = np.random.rand(100, 2) * 0.5
    X = np.concatenate([X_positives, X_negatives])
    y = np.concatenate([np.ones(100), np.zeros(100)])

    print(X.shape, y.shape)
    perceptron = SingleLayerPerceptron(2)
    perceptron.train(X, y, lr=0.01, method="perceptron", epochs=20, visualize=True)



    perceptron2 = SingleLayerPerceptron(2)

    y = np.concatenate([np.ones(100), -np.ones(100)])
    perceptron2.train(X, y, lr=0.003, method="delta", epochs=20, visualize=True)

