# %% RBFClass
# TODO: fix less centers than points in kfold case
# TODO: Make a SpaceTime class, Network + SpaceTime models, adapt LHM methods.

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from pyDOE import lhs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class RBFModel:
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
    
class Kriginig(RBFModel):

    def fit(self):
        """
        Fit a Kriging model to the data using GaussianProcessRegressor.
        """

        # Define the kernel function to be used in Gaussian process
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

        # Initialize the Gaussian Process model
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

        # Fit the GP model to the data
        self.gp.fit(self.X, self.y)

    def predict_kriging(self, Xnew):
        """
        Predict using the Kriging model.

        Parameters:
        Xnew: numpy array
            New points for prediction(n_samples, n_features)

        Returns:
        y_pred: numpy array
            Predicted output (n_samples,)
        """
        y_pred, sigma = self.gp.predict(Xnew, return_std=True)
        return y_pred, sigma


class Preprocessing:

    def kfold(model, k=5, Type='GE', 
              epsi_range = np.logspace(-2,1,50),
              Max_cond = 1e13,
              X_GO=None, y_GO=None, fig=False):
        """
        Complete k-fold hyperparameter selection.

        Parameters:
        model : object
            Instance of the RBF class.
        k : int, optional
            Number of folds. Default is 5.
        model_type : str, optional
            Specify the information used for k-fold. Default is 'GE'.
        X_GO : np.ndarray, optional
            Samples for the Function Value (FV) points for Gradient Only (GO) model.
        y_GO : np.ndarray, optional
            Function values for the GO model.
        fig : bool, optional
            If True, plots the error at each shape factor. Default is False.

        Returns:
        opt_epsi : float
            Optimum shape parameter.
        """
        # Number of samples in each fold
        fold_size_samples = model._n_samples // k
        fold_size_centres = model.C.shape[0] // k

        epsi_range = epsi_range[::-1]

        # Shuffle the data
        indices = np.arange(model._n_samples)
        indicis_centres = np.arange(model.C.shape[0])
        np.random.shuffle(indices)
        np.random.shuffle(indicis_centres)

        e_metrics = np.zeros_like(epsi_range)
        condition = np.zeros_like(epsi_range)
        for cnt, e in enumerate(epsi_range):
            fold_metrics = []

            for i in range(k):
                fold_start_samples = i * fold_size_samples
                fold_end_samples = (i + 1) * fold_size_samples

                fold_indices = indices[fold_start_samples:fold_end_samples]
                train_indices = np.concatenate(
                    (indices[:fold_start_samples], indices[fold_end_samples:]))

                fold_start_centres = i * fold_size_centres
                fold_end_centres = (i + 1) * fold_size_centres

                # centres to train with
                train_indices_centres = np.concatenate(
                    (indicis_centres[:fold_start_centres], indicis_centres[fold_end_centres:]))

                # Separate data into train and validate sets
                X_train = model.X[train_indices, :]
                y_train = model.y[train_indices]
                
                if model.C.shape[0] == model.X.shape[0]:
                    C_train = X_train
                else:
                    C_train = model.C #[train_indices_centres, :]   
                    
                X_valid = model.X[fold_indices, :]
                y_valid = model.y[fold_indices]

                if Type == 'FV':
                    # Create a dummy model to train on training set
                    dummy_model = RBFModel(X_train, y_train, C=C_train)
                    dummy_model.FV_fit(epsi=e)

                    y_pred, dy_pred = dummy_model(X_valid)

                    # Store the RMSE of the fold
                    fold_metrics.append(
                        np.sqrt(np.mean((y_valid - y_pred)**2)))

                elif Type == 'GE':
                    dy_train = model.dy[train_indices, :]
                    dy_valid = model.dy[fold_indices, :]

                    # Scale the func and grad values to have the same influence
                    sf = (np.mean(np.abs(dy_train)) / np.mean(np.abs(y_train)))

                    dummy_model = RBFModel(
                        X_train, y_train, C=C_train, dy=dy_train)
                    
                    dummy_model.GE_fit(epsi=e)

                    y_pred, dy_pred = dummy_model(X_valid)

                    fold_metrics.append(np.sqrt(np.mean((sf*y_valid - sf*y_pred)**2)) +
                                        np.sqrt(np.mean((dy_valid - dy_pred)**2)))

                elif Type == 'GO':
                    dy_train = model.dy[train_indices, :]
                    dy_valid = model.dy[fold_indices, :]

                    dummy_model = RBFModel(
                        X_train, y_train, C=C_train, dy=dy_train)
                    dummy_model.GO_fit(X_GO, y_GO, epsi=e)

                    y_pred, dy_pred = dummy_model(X_valid)

                    fold_metrics.append(
                        np.sqrt(np.mean((dy_valid - dy_pred)**2)))

            # Store the mean RMSE of the folds
            e_metrics[cnt] = np.mean(fold_metrics)
            
            if Type == 'GE':
                model.GE_fit(epsi=e)
                condition[cnt] = model.cond
            else:
                model.FV_fit(epsi=e)
                condition[cnt] = model.cond
            
        epsi_range = epsi_range[condition <= Max_cond]
        e_metrics = e_metrics[condition <= Max_cond]

        if fig:
            plt.figure()
            plt.loglog(epsi_range[::-1], e_metrics[::-1])

        opt_epsi = epsi_range[np.argmin(e_metrics)]

        return opt_epsi
    
    def VaildEpsi(points, actual, model, 
                  Max_cond = 1e12,
                  fig = False,
                  Type = 'GE',
                  epsi_range = np.logspace(-1,1,50)):
        
        error = np.zeros_like(epsi_range)
        condition = np.zeros_like(error)
        for i,e in enumerate(epsi_range):
            
            if Type =='GE':
                model.GE_fit(e)
            else:
                model.FV_fit(e)
                
            error[i] = np.mean((model(points, OnlyFunc = True) - actual)**2)
            condition[i] = model.cond
            
        epsi = epsi_range[np.argmin(error)]
        
        epsi_range = epsi_range[condition <= Max_cond]
        error = error[condition <= Max_cond]

        if fig:
            plt.figure()
            plt.loglog(epsi_range[::-1], error[::-1])

       
        return epsi
    
    def GradientErrors(model, epsi_range = np.logspace(-2,1,100), fig = False, FuncFit = False):
        '''
        Parameters
        ----------
        model : model from RBFModel class.
        epsi_range : np.array, optional
            shape parameters to check. The default is np.logspace(-2,1,100).
        fig : boolen, optional
            plot a error graph. The default is False.
        FuncFit : boolen, optional
            chose to create a functional RBF instead of a gradient enhanced model.

        Returns
        -------
        optimum shape parameter value.

        '''
        
        error = np.zeros_like(epsi_range)
        for i,e in enumerate(epsi_range):
            
            if FuncFit:
                
                model.FV_fit(epsi = e)
                
            else:
                
                model.GE_fit(epsi = e)
            
            error[i] = np.linalg.norm(model.dy_org - model(model.X_org)[1])/np.linalg.norm(model.dy_org)
            #error[i] = np.sum(model.y - model(model.X)) + np.sum((model.dy_org - model(model.X)[1])**2)**0.5
            
        
        opt_epsi = epsi_range[np.argmin(error)]
        
        if FuncFit:
            
            model.FV_fit(epsi = opt_epsi)
            
        else:
            
            model.GE_fit(epsi = opt_epsi)
        
        if fig:
            plt.figure()
            plt.loglog(epsi_range, error)
            plt.loglog(opt_epsi, np.min(error), 'k.')
            
        return opt_epsi
        

    def ASM(model):
        """
        Complete the active subspace method.

        Parameters:
        model : object
            Instance of the RBFModel class.

        Returns:
        c_approx : np.ndarray
            The approximation of the covariance matrix.
            Also adds the eigenvalues and vectors as attributes to the model object.
        """
        C_approx = 0
        for g in model.dy:

            g = g.reshape(-1, 1)  # reshape into a colum vector

            C_approx += g @ g.T  # compute outer product of the vector

        C_approx /= model.dy.shape[0]

        model.eig, model.evec = np.linalg.eigh(C_approx)

        return C_approx

    def GE_LHM(model, n_neighbors=None):
        """
        Computes a Symmetric Rank-One (SR1) Hessian approximation for each point in X
        using the n_neighbors+1 closest points.

        Parameters:
        model : object
            Instance of the RBFModel class.
        n_neighbors : int, optional
            Number of neighbors. Default is None, which sets it to n_features + 1.

        Returns:
        h_mean : np.ndarray
            The approximation of the average local Hessian.
            Also adds the eigenvalues and vectors as attributes to the model object.
        """

        n_samples, n_features = model.X.shape
        if n_neighbors is None:
            n_neighbors = n_features + 1

        # Initialize the SR1 Hessian approximations to the identity matrix
        hessian = np.zeros(
            (model._n_samples, model._n_features, model._n_features))
        for i in range(model._n_samples):
            hessian[i] = np.eye(model._n_features)

        # Compute the SR1 Hessian approximation for each point
        dist_matrix = cdist(model.X, model.X, metric='euclidean')
        closest_indices = np.argsort(dist_matrix, axis=1)[:, 1:n_neighbors+1]

        for i in range(model._n_samples):

            hessian[i] = model.dy[[i], :].T @ model.dy[[i], :]

            for j in closest_indices[i][::-1]:

                DeltaX = model.X[[j], :].T - model.X[[i], :].T

                DeltaJac = model.dy[[j], :].T - model.dy[[i], :].T

                Term = (DeltaJac - hessian[i] @ DeltaX)
                
                if ((Term > -1e-6) & (Term < 1e-6)).any():
                    
                    break
                
                hessian[i] += (Term @ Term.T)/(Term.T @ DeltaX)

        for i, h in enumerate(hessian):

            eig, evec = np.linalg.eig(h)
            
            if np.any(np.iscomplex(eig)):

                h_new = np.identity(n_features)

            else:

                h_new = evec @ np.diag(abs(eig)) @ evec.T

            hessian[i] = h_new

        H_mean = np.median(np.round(hessian, 10), axis=0)

        model.eig, model.evec = np.linalg.eig(H_mean)
        # print(model.eig)
        return H_mean

    def FV_LHM(model):

        from sklearn.metrics.pairwise import euclidean_distances
        from itertools import combinations

        n_points, n_features = model.X.shape
        H = np.zeros((n_points, n_features, n_features))

        k_neighbors = 1 + n_features + n_features * (n_features - 1) // 2
        k_neighbors = int(n_features*(n_features + 1)/2 + n_features + 1)

        k_neighbors = len(np.triu_indices(n_features)[0]) + n_features + 1

        for i in range(n_points):
            x = model.X[i]
            f = model.y[i]

            # Calculate pairwise Euclidean distances
            distances = euclidean_distances(model.X, [x]).flatten()

            # Sort distances and get the indices of the k_neighbors closest points
            closest_indices = np.argsort(distances)[:k_neighbors]

            # Extract the closest points and function values
            closest_points = model.X[closest_indices]
            closest_values = model.y[closest_indices]

            # Fit local quadratic model using the closest points
            A = np.ones((k_neighbors, k_neighbors))

            A[:, :n_features] = (closest_points - x)**2  # square terms

            for j in range(k_neighbors):
                A[j, n_features:n_features+len(np.triu_indices(n_features, k=1)[0])] = [
                    np.product(k) for k in combinations(closest_points[j] - x, 2)]

            A[:, n_features+len(np.triu_indices(n_features, k=1)[0]):-
              1] = closest_points - x  # linear terms

            b = closest_values - f

            # Solve the linear system to obtain the local quadratic coefficients
            coeff = np.linalg.solve(A, b)

            # Build the Hessian matrix using the quadratic coefficients
            h = np.zeros((n_features, n_features))
            h[np.diag_indices(n_features)] = coeff[:n_features, 0] * 2

            h[np.triu_indices(n_features, k=1)] = coeff[n_features:
                                                        n_features+len(np.triu_indices(n_features, k=1)[0]), 0]

            h[np.tril_indices(n_features, k=-1)
              ] = h[np.triu_indices(n_features, k=1)]

            H[i, :, :] = h

        for k in range(n_points):

            h = H[k, :, :]

            eig, evec = np.linalg.eig(h)

            if np.any(np.iscomplex(eig)):

                H[k, :, :] = np.identity(n_features)

            else:

                H[k, :, :] = evec @ np.diag(abs(eig)) @ evec.T

        Ha = np.median(H, axis=0)

        model.eig, model.evec = np.linalg.eigh(Ha)
        model.eig /= model.eig[0]

        return Ha
    
    def SP_LHM(model, SpaceSamples, min_step, n_neighbors=None):
        '''
        Complete the LHM method in the space-time sampling case.
        The 'time' domain is scaled for a better Hessain approximation.
        
        Parameters
        ----------
        SpaceSamples: The samples locations in the space domain.
        min_step: Minium, step the solver can take
        n_neighbors : TYPE, optional
            DESCRIPTION. The default is None.
            

        Returns
        -------
        H: TYPE
            DESCRIPTION.

        '''
        
        dist_matrix = cdist(SpaceSamples, SpaceSamples, metric='euclidean')
        dist_matrix = np.sort(dist_matrix, axis=1)
        
        c = np.mean(dist_matrix[:,SpaceSamples.shape[1]+1])
        #c = np.mean(dist_matrix[:,2])
        
        s = np.ones(model.X.shape[1])
        
        s[-1] = c/(min_step)
        
        n_samples, n_features = model.X.shape
        if n_neighbors is None:
            n_neighbors = n_features + 1
        
        # Initialize the SR1 Hessian approximations to the identity matrix
        hessian = np.zeros(
            (model._n_samples, model._n_features, model._n_features))
        for i in range(model._n_samples):
            hessian[i] = np.eye(model._n_features)
        
        # Compute the SR1 Hessian approximation for each point
        dist_matrix = cdist(model.X * s, model.X * s, metric='euclidean')
        closest_indices = np.argsort(dist_matrix, axis=1)[:, 1:n_neighbors+1]
        
        '''
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(*model.X.T, 'k.')
        
        for i in np.random.randint(0, model.X.shape[0], size = (4)):
            
            for j in closest_indices[i][::-1]:
                
                ax.plot([model.X[i, 0], model.X[j, 0]],
                        [model.X[i, 1], model.X[j, 1]], 
                        [model.X[i, 2], model.X[j, 2]], 'r--')
              '''
        
        for i in range(model._n_samples):
        
            hessian[i] = model.dy[[i], :].T @ model.dy[[i], :]
        
            for j in closest_indices[i][::-1]:
        
                DeltaX = model.X[[j], :].T - model.X[[i], :].T
        
                DeltaJac = model.dy[[j], :].T - model.dy[[i], :].T
        
                Term = (DeltaJac - hessian[i] @ DeltaX)
                
                if ((Term > -1e-6) & (Term < 1e-6)).any():
                    
                    break
                #DeltaX[DeltaX == 0] = 1
                hessian[i] += (Term @ Term.T)/(Term.T @ DeltaX)
        
        for i, h in enumerate(hessian):
            
            eig, evec = np.linalg.eig(h)
        
            if np.any(np.iscomplex(eig)):

                hessian[i] = np.identity(n_features)

            else:

                hessian[i] = evec @ np.diag(abs(eig)) @ evec.T
        
        H_mean = np.median(np.round(hessian, 10), axis=0)
        
        model.eig, model.evec = np.linalg.eig(H_mean)
        # print(model.eig)
        return H_mean


    def Transform(model, method='GE-LHM', n_neighbors=None, 
                                          R=None, S=None,
                                          space_samples=None, min_step=None):
        """
        Transforms the model according to a chosen method.

        Parameters:
        model : object
            Instance of the RBFModel class.
        method : str, optional
            Which transformation method to use. Default is 'GE-LHM'.
        n_neighbors : int, optional
            Neighbours for LHM methods. Default is None.
        R : np.ndarray, optional
            User-defined rotation matrix. Default is None.
        S : np.ndarray, optional
            User-defined scalers. Default is None.

        Returns:
        None.
        """

        if method == 'GE-LHM':
            Preprocessing.GE_LHM(model, n_neighbors=n_neighbors)
        
        if method == 'SP-GE-LHM':
            Preprocessing.SP_LHM(model, space_samples, min_step, n_neighbors=n_neighbors)

        if method == 'GE-DLHM':
            H = Preprocessing.GE_LHM(model, n_neighbors=n_neighbors)

            model.eig, model.evec = np.diag(H), np.identity(model._n_features)

        if method == 'FV-LHM':
            Preprocessing.FV_LHM(model)

        if method == 'ASM':
            Preprocessing.ASM(model)

        if method == 'ideal':
            model.evec = R
            model.eig = S
            
        model.scalers = model.eig**0.5 
        #model.scalers /= model.scalers[-1]
 
        model.X_org = np.copy(model.X)
        model.X = model.X @ model.evec * model.scalers

        model.C_org = np.copy(model.C)
        model.C = model.C @ model.evec * model.scalers

        if model.dy is not None:
            model.dy_org = np.copy(model.dy)
            model.dy = model.dy @ model.evec / model.scalers

        return

# %%% Test code

# Define the actual function to be modeled


if __name__ == '__main__':

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
    model = RBFModel(train_X, train_Y, dy=train_dy)

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
    
#%%






