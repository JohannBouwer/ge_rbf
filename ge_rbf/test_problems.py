import numpy as np

def NonIso(X, R=None):
    '''
    Function that purposefully has ideal scaling in each direction.
    
    Parameters
    ----------
    X : numpy array
        data points (n_samples, n_features).
    R : numpyy array, optional
        orthagonal matrix to rotate the domain. The default is None.
        None: random rotation
        0: No rotation

    Returns
    -------
    func : numpy array
        function vaules (n_samples, 1).
    grads : numpy array
        gradient vaules (n_samples, n_features).

    '''
    n = X.shape[1]

    if R == None:
        R = np.identity(n)

    dimensions = np.arange(1, n + 1, 1).reshape(1, -1)

    Freqs = (1.5*np.pi)/(1 + np.exp((10)*(-dimensions + n/2))) + np.pi/2

    Amps = -(2)*np.exp(-(2/n)*(dimensions - n/2)**2) + 3

    func = (1/n)*np.sum(Amps*np.sin(Freqs * X), axis=1).reshape(-1, 1)

    grads = (1/n)*Amps*Freqs*np.cos(Freqs * X)

    grads = grads.dot(R)

    return func, grads


def Rosenbach(X):
    '''
    Commonly used  Rosenbach testing function.
    
    Parameters
    ----------
    X : numpy array
        data points (n_samples, n_features).

    Returns
    -------
    func : numpy array
        function vaules (n_samples, 1).
    grads : numpy array
        gradient vaules (n_samples, n_features).

    '''
    n = X.shape[1]

    func = np.zeros(X.shape[0])

    for i in range(X.shape[0]):

        func[i] = sum(100 * (X[i, j+1] - X[i, j]**2)**2 +
                      (1 - X[i, j])**2 for j in range(n-1))

    grads = np.zeros((X.shape[0], n))

    for i in range(X.shape[0]):

        for j in range(n-1):

            grads[i, j] += -400 * X[i, j] * \
                (X[i, j+1] - X[i, j]**2) - 2 * (1 - X[i, j])
            grads[i, j+1] += 200 * (X[i, j+1] - X[i, j]**2)

    return func.reshape(-1,1), grads

def rastrigin(X):
    """
    Rastrigin function and its gradient for multiple sample points.
    
    Parameters:
    X : ndarray of shape (n_samples, n_features)
        Input sample points.
    
    Returns:
    func : ndarray of shape (n_samples,)
        Function values at input points.
    grad : ndarray of shape (n_samples, n_features)
        Gradient values at input points.
    """
    X = np.atleast_2d(X)
    A = 10
    
    func = A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    
    grad = 2 * X + 2 * np.pi * A * np.sin(2 * np.pi * X)
    
    return func, grad

def ackley(X):
    """
    Ackley function and its gradient for multiple sample points.
    
    Parameters:
    X : ndarray of shape (n_samples, n_features)
        Input sample points.
    
    Returns:
    f : ndarray of shape (n_samples,)
        Function values at input points.
    grad : ndarray of shape (n_samples, n_features)
        Gradient values at input points.
    """
    X = np.atleast_2d(X)
    a, b, c = 20, 0.2, 2 * np.pi
    d = X.shape[1]
    
    sum1 = np.sum(X**2, axis=1)
    sum2 = np.sum(np.cos(c * X), axis=1)
    
    func = -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e
    
    grad = ((a * b / d) * np.exp(-b * np.sqrt(sum1 / d))[:, None] * (2 * X) + 
            (c / d) * np.exp(sum2 / d)[:, None] * np.sin(c * X))
    
    return func, grad

def sphere(X):
    """
    Sphere function and its gradient for multiple sample points.
    
    Parameters:
    X : ndarray of shape (n_samples, n_features)
        Input sample points.
    
    Returns:
    func : ndarray of shape (n_samples,)
        Function values at input points.
    grad : ndarray of shape (n_samples, n_features)
        Gradient values at input points.
    """
    X = np.atleast_2d(X)
    
    func = np.sum(X**2, axis=1)
    
    grad = 2 * X
    
    return func, grad

def beale(X):
    """
    Beale function (for 2D) and its gradient for multiple sample points.
    
    Parameters:
    X : ndarray of shape (n_samples, 2)
        Input sample points (only 2D inputs are valid).
    
    Returns:
    func : ndarray of shape (n_samples,)
        Function values at input points.
    grad : ndarray of shape (n_samples, 2)
        Gradient values at input points.
    """
    X = np.atleast_2d(X)
    if X.shape[1] != 2:
        raise ValueError("Beale function is only defined for 2 variables.")
        
    x1, x2 = X[:, 0], X[:, 1]
    func = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2**2) ** 2 + (2.625 - x1 + x1 * x2**3) ** 2
    
    df_dx1 = 2 * (1.5 - x1 + x1 * x2) * (-1 + x2) + 2 * (2.25 - x1 + x1 * x2**2) * (-1 + x2**2) + 2 * (2.625 - x1 + x1 * x2**3) * (-1 + x2**3)
    df_dx2 = 2 * (1.5 - x1 + x1 * x2) * x1 + 2 * (2.25 - x1 + x1 * x2**2) * (2 * x1 * x2) + 2 * (2.625 - x1 + x1 * x2**3) * (3 * x1 * x2**2)
    grad = np.column_stack([df_dx1, df_dx2])
    
    return func, grad
