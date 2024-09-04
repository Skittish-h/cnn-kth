import numpy as np
import matplotlib.pyplot as plt


def generate_points_normal(n_points, mean, std, label):
    # Generate points and their corresponding labels as tuples
    # mean is [mean_x, mean_y]
    points_with_labels = [
        ([np.random.normal(mean[0], std), np.random.normal(mean[1], std)], label)
        for _ in range(n_points)
    ]
    return points_with_labels


def shuffle_points(points):
    np.random.shuffle(points)
    return points


def plot_points_2d(points, title=""):
    # Plot the points
    for point, label in points:
        if (label == 1):  # positive
            plt.scatter(point[0], point[1], c='r')
        else:  # negative
            plt.scatter(point[0], point[1], c="b")
    plt.title(title)
    plt.show()
