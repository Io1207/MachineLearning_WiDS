import numpy as np
import matplotlib.pyplot as plt

def hypothesis(x, theta0, theta1):
    return theta0 + theta1 * x

def compute_cost(x, y, theta0, theta1):
    m = len(y)
    total_error = 0.0
    for i in range(m):
        total_error += (hypothesis(x[i], theta0, theta1) - y[i]) ** 2
    return total_error / (2 * m)

# Gradient Descent function
def gradient_descent(x, y, theta0, theta1, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        sum_error0 = 0
        sum_error1 = 0
        
        #gradients for theta0 and theta1
        for i in range(m):
            sum_error0 += (hypothesis(x[i], theta0, theta1) - y[i])
            sum_error1 += (hypothesis(x[i], theta0, theta1) - y[i]) * x[i]
        

        theta0 -= alpha * sum_error0 / m
        theta1 -= alpha * sum_error1 / m

    return theta0, theta1

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 3.5, 5])
    theta0 = 0.0 
    theta1 = 0.0
    alpha = 0.01 
    iterations = 1000  
    print("a_2 x+ a_1")
    initial_cost = compute_cost(x, y, theta0, theta1)
    print(f"Initial cost: {initial_cost}")

    theta0, theta1 = gradient_descent(x, y, theta0, theta1, alpha, iterations)

    print(f"Optimized a1: {theta0}")
    print(f"Optimized a2: {theta1}")
    final_cost = compute_cost(x, y, theta0, theta1)
    print(f"Final cost: {final_cost}")
    plt.scatter(x, y, color="red", marker="x", label="Training Data")
    plt.plot(x, hypothesis(x, theta0, theta1), label="Linear Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
