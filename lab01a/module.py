import numpy as np
import matplotlib.pyplot as plt


class SingleLayerPerceptron:
    def __init__(self, n, no_bias=False):
        # single layer perceptron with n inputs
        # bias is the last element of w
        self.no_bias = no_bias
        if no_bias:
            self.w = np.random.randn(n)
        else:
            self.w = np.random.randn(n + 1)  # Initialize with random values including negative ones

    def forward(self, x):
        # forward pass
        if self.no_bias:
            return np.dot(self.w, x)
        return np.dot(self.w, np.append(x, 1.0))

    def _update(self, x, y, lr, method="delta"): # note delta expects y to be -1 or 1 and perceptron expects 0 or 1
        dw = np.zeros_like(self.w)
        output = self.forward(x)
        # update the weights
        if method == "delta":
            error = y - output
            inp = np.append(x, 1.0) if not self.no_bias else x
            dw = lr * error * inp
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
        x1_vals = np.array([min(X[:, 0]) - 1, max(X[:, 0]) + 1])

        # Compute the corresponding x2 values using the decision boundary formula
        # Assuming self.w[0] is w1, self.w[1] is w2, and self.w[2] is the bias term b
        if self.no_bias:
            x2_vals = -(self.w[0] * x1_vals) / self.w[1]
        else:
            x2_vals = -(self.w[0] * x1_vals + self.w[2]) / self.w[1]

        # Plot the decision boundary
        plt.plot(x1_vals, x2_vals, 'k--', label="Decision Boundary")
        plt.show()

    def train(self, X, y, lr, method="delta", epochs=10, visualize=False, batch=True):
        current_loss = self.get_loss(X, y)
        loss = [current_loss]

        if visualize:
            self.visualize(X, y)

        for i in range(epochs):
            dw = np.zeros_like(self.w)
            for j in range(len(X)):
                n_dw = self._update(X[j], y[j], lr, method)
                dw += n_dw
                # if online learning, update the weights after each sample
                if not batch:
                    self.w += n_dw
            # if batch learning, update the weights after each epoch
            if batch:
                self.w += dw
            if visualize:
                self.visualize(X, y)
            mse, misclassified = self.get_loss(X, y)
            loss.append((mse, misclassified))

        return loss

    def get_loss(self, X, y, method="delta"):
        mse = 0
        misclassified = 0

        # Loop through each sample
        for i in range(len(X)):
            output = self.forward(X[i])
            if method == "delta":
                mse += (y[i] - output) ** 2
                if np.sign(y[i]) != np.sign(output):
                    misclassified += 1

            # Perceptron loss calculation (classification errors)
            elif method == "perceptron":
                if y[i] == 1 and output < 0:
                    misclassified += 1
                elif y[i] == -1 and output >= 0:
                    misclassified += 1

        mse /= len(X)
        return mse, misclassified

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
