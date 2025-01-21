import numpy as np
import matplotlib.pyplot as plt
from rbfmodels import RBFmodel
from preprocessing import Preprocessing
from pyDOE import lhs

def actual_function(x, y):
    return np.sin(0.1*np.pi * x) + np.sin(np.pi * y)

def grad_function(x, y):
    dx = 0.1*np.pi*np.cos(0.1*np.pi*x).reshape(-1, 1)
    dy = np.pi*np.cos(np.pi*y).reshape(-1, 1)
    return np.hstack((dx, dy))

# Create a meshgrid of x and y values
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# Calculate the corresponding z values using the actual function
Z_actual = actual_function(X, Y)
dZ_actual = grad_function(X, Y)

# Train the model using some training data
train_X = lhs(2, samples=10, criterion='m')*2.1 - 1.05
 
train_Y = actual_function(train_X[:, 0], train_X[:, 1]).reshape(-1, 1)
train_dy = grad_function(train_X[:, 0], train_X[:, 1])

# Create an instance of the RBFModel clas
#train_dy = None
model = RBFmodel(train_X, train_Y, dy=train_dy)

Preprocessing.Transform(model, method='GE-LHM')


opt_epsi = Preprocessing.GradientErrors(model, epsi_range = np.logspace(-1,1,100), fig=True)

#model.GO_fit(train_X[[2], :], train_Y[2], epsi=opt_epsi)
model.GE_fit(epsi=opt_epsi)
# model.FV_fit(epsi=opt_epsi)

# Generate predictions using the trained model
Z_predicted = model(np.column_stack((X.flatten(), Y.flatten())))
Z_predicted, dZ_predicted = Z_predicted[0].reshape(X.shape), Z_predicted[1]

error = np.round(np.mean((Z_actual - Z_predicted)**2), 4)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the actual function
ax.plot_surface(X, Y, Z_actual, alpha=0.5)

# Plot the RBFModel predictions
ax.plot_surface(X, Y, Z_predicted, alpha=0.8)

# Set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Error {error}')

# Show the plot
plt.show()

