import numpy as np
import matplotlib.pyplot as plt


def generate_points_uniform(n_points, n_features, low, high, label):
    # Generate points and their corresponding labels as tuples
    points_with_labels = [
        (np.random.uniform(low, high, n_features), label)
        for _ in range(n_points)
    ]
    return points_with_labels


def generate_points_normal(n_points, n_features, mean, std, label):
    # Generate points and their corresponding labels as tuples
    points_with_labels = [
        (np.random.normal(mean, std, n_features), label)
        for _ in range(n_points)
    ]
    return points_with_labels


def shuffle_points(points):
    np.random.shuffle(points)
    return points


def plot_points(points):
    # Plot the points
    for point, label in points:
        plt.scatter(point[0], point[1], c=label)
    plt.show()
