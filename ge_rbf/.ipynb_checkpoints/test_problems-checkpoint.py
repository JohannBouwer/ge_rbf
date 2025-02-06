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
    Commonly used testing function.
    
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
