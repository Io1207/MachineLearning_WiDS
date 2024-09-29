#As I wrote this code, I was heavily referring to geeksforgeeks
#there might be some very, very similar snippets
import numpy as np

class SGD:
    def __init__(self, learn=0.01, epochs=1000, batchSize=32, tol=0.001):
        self.learning_rate = learn
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.batchSize = batchSize
        self.tolerance = tol

    
    def fit(self, x, y):
        rows=x.shape[0]
        cols = x.shape[1]
        self.bias = np.random.randn()
        self.weights = np.random.randn(cols)

        for iter in range(self.epochs):
            indices = np.random.permutation(rows)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for i in range(0, rows, self.batchSize):
                x_batch = x_shuffled[i:i+self.batchSize]
                y_batch = y_shuffled[i:i+self.batchSize]

                weight, bias = self.gradient(x_batch, y_batch)
                self.weights -= self.learning_rate * weight
                self.bias -= self.learning_rate * bias

            if iter % 100 == 0:
                y_pred = self.predict(x)
                loss = self.meanSqError(y, y_pred)
                print(f"Epoch {iter}: Loss {loss}")

            if np.linalg.norm(weight) < self.tolerance:
                print("Converged")
                break
        return self.weights, self.bias    

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

    def meanSqError(self, y_true, y_pred):
        return np.mean((y_true - y_pred)*(y_true - y_pred))

    def gradient(self, x_batch, y_batch):
        y_pred = self.predict(x_batch)
        error = y_pred - y_batch
        weight = np.dot(x_batch.T, error) / x_batch.shape[0]
        bias = np.mean(error)
        return weight, bias

    
x = np.random.randn(100, 5)
y = np.dot(x, np.array([1, 2, 3, 4, 5]))\
    + np.random.randn(100) * 0.1
model = SGD(learn=0.01, epochs=1000,
            batchSize=32, tol=1e-3)
w,b=model.fit(x,y)
y_pred = w*x+b


#I have not written the code after this point, I came from geeksforgeeks
#I do not understand the code, please don't think that I wrote it. 
#This is a property of user rahul_roy on geeksforgeeks
#---------------------------------------------------------------

# import tensorflow as tf

# class SGD:
#     def __init__(self, lr=0.001, epochs=2000, batch_size=32, tol=1e-3):
#         self.learning_rate = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.tolerance = tol
#         self.weights = None
#         self.bias = None

#     def predict(self, X):
#         return tf.matmul(X, self.weights) + self.bias

#     def mean_squared_error(self, y_true, y_pred):
#         return tf.reduce_mean(tf.square(y_true - y_pred))

#     def gradient(self, X_batch, y_batch):
#         with tf.GradientTape() as tape:
#             y_pred = self.predict(X_batch)
#             loss = self.mean_squared_error(y_batch, y_pred)
#         gradient_weights, gradient_bias = tape.gradient(loss, [self.weights, self.bias])
#         return gradient_weights, gradient_bias

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = tf.Variable(tf.random.normal((n_features, 1)))
#         self.bias = tf.Variable(tf.random.normal(()))

#         for epoch in range(self.epochs):
#             indices = tf.random.shuffle(tf.range(n_samples))
#             X_shuffled = tf.gather(X, indices)
#             y_shuffled = tf.gather(y, indices)

#             for i in range(0, n_samples, self.batch_size):
#                 X_batch = X_shuffled[i:i+self.batch_size]
#                 y_batch = y_shuffled[i:i+self.batch_size]

#                 gradient_weights, gradient_bias = self.gradient(X_batch, y_batch)
#                 # Gradient clipping
#                 gradient_weights = tf.clip_by_value(gradient_weights, -1, 1)
#                 gradient_bias = tf.clip_by_value(gradient_bias, -1, 1)
                
#                 self.weights.assign_sub(self.learning_rate * gradient_weights)
#                 self.bias.assign_sub(self.learning_rate * gradient_bias)

#             if epoch % 100 == 0:
#                 y_pred = self.predict(X)
#                 loss = self.mean_squared_error(y, y_pred)
#                 print(f"Epoch {epoch}: Loss {loss}")

#             if tf.norm(gradient_weights) < self.tolerance:
#                 print("Convergence reached.")
#                 break

#         return self.weights.numpy(), self.bias.numpy()

# # Create random dataset with 100 rows and 5 columns
# X = np.random.randn(100, 5).astype(np.float32)
# # Create corresponding target value by adding random
# # noise in the dataset
# y = np.dot(X, np.array([1, 2, 3, 4, 5], dtype=np.float32)) + np.random.randn(100).astype(np.float32) * 0.1

# # Create an instance of the SGD class
# model = SGD(lr=0.005, epochs=1000, batch_size=12, tol=1e-3)
# w, b = model.fit(X, y)

# # Predict using predict method from model
# y_pred = np.dot(X, w) + b