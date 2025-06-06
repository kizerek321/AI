import numpy as np

# Step function (binary activation)
def step(x):
    return 1 if x > 0 else 0

# Neuron class
def perceptron(weights, bias, inputs):
    x = np.dot(weights, inputs) + bias
    return step(x)

# Final network
def classify_point(x, y):
    inputs = np.array([x, y])

    # First layer: 3 perceptrons
    p1 = perceptron(weights=np.array([-1, 0]), bias=3, inputs=inputs)  # x < 3
    p2 = perceptron(weights=np.array([0, -1]), bias=2, inputs=inputs)  # y < 2
    p3 = perceptron(weights=np.array([-1, -1]), bias=4, inputs=inputs) # x + y < 4

    # Second layer: AND of outputs from p1, p2, p3
    and_input = np.array([p1, p2, p3])
    final = perceptron(weights=np.array([1, 1, 1]), bias=-2.5, inputs=and_input)

    return final  # 1 = class 1, 0 = class 0


if __name__ == '__main__':
    print(classify_point(1.5, 1.5))  # Should return 1 (class 1)
    print(classify_point(3.1, 1.0))  # Should return 0 (class 0)
    print(classify_point(2.0, 2.1))  # Should return 0 (class 0)
    print(classify_point(2.5, 1.0))  # Should return 1 (class 1)
