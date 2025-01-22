
import numpy as np
from scipy.spatial.distance import cdist


class RBFmodel:
    """
    A class to implement Radial Basis Function (RBF) models including 
    Function Value (FV), Gradient Enhanced (GE), and Gradient Only (GO) models.
    """

    def __init__(self, X, y, C=None, dy=None):
        """
        Constructor for the RBF model.

        Parameters:
        X : np.ndarray
            Input data with shape (n_samples, n_features).
        y : np.ndarray
            Target data with shape (n_samples, ).
        C : np.ndarray, optional
            Centres for the RBF model, default is X. Shape: (n_centres, n_features).
        dy : np.ndarray, optional
            Gradient values with shape (n_samples, n_features).
        """
        self.X = np.copy(X)
        self.X_org = np.copy(X)
        self.y = np.copy(y)
        self.dy = np.copy(dy)
        self.dy_org = None if dy is None else np.copy(dy)
        self.C = np.copy(X) if C is None else np.copy(C)

        self._n_samples, self._n_features = self.X.shape
        self._n_centres = self.C.shape[0]

        self.eig = np.ones(self._n_features)
        self.scalers = np.copy(self.eig)
        self.evec = np.identity(self._n_features)

    def FV_fit(self, epsi=1):
        """
        Fit the Function Value Radial Basis Function (FV-RBF) model.

        Parameters:
        epsi : float, optional
            The parameter for the RBF. Default is 1.
        """
        self.epsi = epsi

        # Calculate the distance between each pair of points
        dist_matrix = cdist(self.X, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Solve the linear system to find the coefficients
        self.coefficients = np.linalg.solve(kernel_matrix, self.y)

        self.cond = np.linalg.cond(kernel_matrix)

    def GE_fit(self, epsi=1):
        """
        Fit the Gradient Enhanced Radial Basis Function (GE-RBF) model.

        Parameters:
        epsi : float, optional
            The parameter for the RBF. Default is 1.
        """
        self.epsi = epsi

        # Calculate the distance between each pair of points
        dist_matrix = cdist(self.X, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Calculate the RBF gradient kernel matrix (n_samples, n_samples, n_feat)
        Xm = np.repeat(self.X[:, None, :], self._n_centres, axis=1)
        Cm = np.repeat(self.C[None, :, :], self._n_samples, axis=0)

        XC = (Xm - Cm).transpose(2, 0, 1)

        kernel_gradients = -2 * self.epsi * XC * kernel_matrix[None, :, :]

        # Reshape the kernel gradient matrix
        kernel_gradients = kernel_gradients.reshape(
            self._n_samples*self._n_features, self._n_centres)

        # Create the full matrix
        kernel_matrix = np.vstack((kernel_matrix, kernel_gradients))

        # Least squares fit
        Y = np.vstack((self.y.reshape(-1, 1),
                       self.dy.reshape(self._n_samples * self._n_features, 1, order='F')))

        #self.coefficients = np.linalg.solve(
            #kernel_matrix.T @ kernel_matrix, kernel_matrix.T @ Y)
        
        self.coefficients = np.linalg.lstsq(kernel_matrix, Y, rcond=None)[0]

        self.cond = np.linalg.cond(kernel_matrix)

    def GO_fit(self, X_FV, y, epsi=1):
        """
        Fit the Gradient Only Radial Basis Function (GO-RBF) model.

        Parameters:
        X_FV : np.ndarray
            Sample locations for the function value.
        y : np.ndarray
            Function values.
        epsi : float, optional
            The parameter for the RBF. Default is 1.
        """
        self.epsi = epsi

        # Calculate the kernel gradient matrix
        dist_matrix = cdist(self.X, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Calculate the RBF gradient kernel matrix (n_samples, n_samples, n_feat)
        Xm = np.repeat(self.X[:, None, :], self._n_centres, axis=1)
        Cm = np.repeat(self.C[None, :, :], self._n_samples, axis=0)

        XC = (Xm - Cm).transpose(2, 0, 1)

        kernel_gradients = -2 * self.epsi * XC * kernel_matrix[None, :, :]

        # Reshape the kernel gradient matrix
        kernel_gradients = kernel_gradients.reshape(
            self._n_samples*self._n_features, self._n_centres)

        dist_matrix_FV = cdist(X_FV, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix for the function value points
        kernel_matrix_FV = np.exp(-self.epsi * (dist_matrix_FV ** 2))

        # Create the full matrix
        kernel_matrix = np.vstack((kernel_matrix_FV, kernel_gradients))

        # Least squares fit
        Y = np.vstack((y.reshape(-1, 1),
                       self.dy.reshape(self._n_samples * self._n_features, 1, order='F')))

        self.coefficients = np.linalg.solve(
            kernel_matrix.T @ kernel_matrix, kernel_matrix.T @ Y)

        return

    def __call__(self, Xnew, OnlyFunc=False):
        """
        Predict using the RBF model.

        Parameters:
        Xnew : np.ndarray
           New points for prediction. Shape: (n_samples, n_features).
        OnlyFunc : bool, optional
           If True, only function value is returned. Default is False.

        Returns:
        y_pred : np.ndarray
            Predicted output. Shape: (n_samples, ).
        dy_pred : np.ndarray
            Predicted gradients. Shape: (n_samples, n_features).
        """
        # Transform the domain
        Xt = Xnew @ self.evec * self.scalers

        # Calculate the distance between each input point and the training points
        dist_matrix = cdist(Xt, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Calculate the predicted output
        y_pred = kernel_matrix @ self.coefficients

        if OnlyFunc:
            return y_pred

        # Calculate the gradient matrix
        Xm = np.repeat(Xt[:, None, :], self._n_centres, axis=1)
        Cm = np.repeat(self.C[None, :, :], Xt.shape[0], axis=0)

        XC = (Xm - Cm).transpose(2, 0, 1)

        kernel_gradients = -2 * self.epsi * XC * kernel_matrix[None, :, :]

        # Reshape the kernel gradient matrix
        kernel_gradients = kernel_gradients.reshape(
            Xt.shape[0]*self._n_features, self._n_centres)

        # chain rule including the rotation.
        dy_pred = ((kernel_gradients @
                   self.coefficients).reshape(Xnew.shape, order='F') * self.scalers) @ (self.evec.T )

        return y_pred, dy_pred








