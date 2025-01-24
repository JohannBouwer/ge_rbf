import numpy as np
import matplotlib.pyplot as plt
from rbf_models import RBFmodel
from scipy.spatial.distance import cdist

class HyperParameterSelection:
    '''
    Methods to select the shape parameter for the rbf model.
    '''
    
    @staticmethod
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
                    dummy_model = RBFmodel(X_train, y_train, C=C_train)
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

                    dummy_model = RBFmodel(
                        X_train, y_train, C=C_train, dy=dy_train)
                    
                    dummy_model.GE_fit(epsi=e)

                    y_pred, dy_pred = dummy_model(X_valid)

                    fold_metrics.append(np.sqrt(np.mean((sf*y_valid - sf*y_pred)**2)) +
                                        np.sqrt(np.mean((dy_valid - dy_pred)**2)))

                elif Type == 'GO':
                    dy_train = model.dy[train_indices, :]
                    dy_valid = model.dy[fold_indices, :]

                    dummy_model = RBFmodel(
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
        
        opt_epsi = epsi_range[np.argmin(e_metrics)]

        #plot error vs shape parameter
        if fig:
            plt.figure()
            plt.loglog(epsi_range[::-1], e_metrics[::-1])
            plt.xlabel('Shape Parameter')
            plt.ylabel('Relative Error')
            plt.title('Log of Error vs Shape Parameter')
            plt.loglog(opt_epsi, np.min(e_metrics), 'k.')

        return opt_epsi
    
    @staticmethod
    def validation_set(model, Xvaild, y_actual,
                  Max_cond = 1e12,
                  fig = False,
                  Type = 'GE',
                  epsi_range = np.logspace(-1,1,50)):
        '''
        Parameters
        ----------
        model : rbf_model instance
            instance of the rbf model.
        Xvaild : numpy array
            locations of the validation set in the design space.
        y_actual : numpy array
            function target values for the model.
        Max_cond : float, optional
            max conditional value of the kernel matrix. The default is 1e12.
        fig : boolean, optional
            plot the error vs shape parameter (log domain). The default is False.
        Type : string, optional
            What type of model to fit. The default is 'GE'. Options are FV, GE.
        epsi_range : numpy array, optional
            shape aprameter values to trail. The default is np.logspace(-1,1,50).

        Returns
        -------
        epsi : float
            Optimial shape parameter for the model.

        '''
        
        error = np.zeros_like(epsi_range)
        condition = np.zeros_like(error)
        #calculate error for each shape parameter
        for i,e in enumerate(epsi_range):
            
            if Type =='GE':
                
                model.GE_fit(e)
                
            if Type == 'FV':
                
                model.FV_fit(e)
            
            error[i] = np.linalg.norm(model(Xvaild, OnlyFunc = True) - y_actual)/np.linalg.norm(y_actual)
            condition[i] = model.cond
            
        opt_epsi = epsi_range[np.argmin(error)]
        #train the model with the optimin shape parameter
        if Type =='GE':
            
            model.GE_fit(opt_epsi)
            
        if Type == 'FV':
            
            model.FV_fit(opt_epsi)
        
        epsi_range = epsi_range[condition <= Max_cond]
        error = error[condition <= Max_cond]
        #plot error vs shape parameter
        if fig:
            plt.figure()
            plt.loglog(epsi_range[::-1], error[::-1])
            plt.xlabel('Shape Parameter')
            plt.ylabel('Relative Error')
            plt.title('Log of Error vs Shape Parameter')
            plt.loglog(opt_epsi, np.min(error), 'k.')

        return opt_epsi
    
    @staticmethod
    def gradient_validation(model, epsi_range = np.logspace(-2,1,100), fig = False, Type = 'GE'):
        '''
        Parameters
        ----------
        model : rbf_model instance
            instance of the rbf model.
        epsi_range : numpy array, optional
            shape parameters to check. The default is np.logspace(-2,1,100).
        fig : boolen, optional
            plot a error graph. The default is False.
        FuncFit : boolen, optional
            chose to create a functional RBF instead of a gradient enhanced model.

        Returns
        -------
        optimum shape parameter value.

        '''
        #calculate error for each shape parameter
        error = np.zeros_like(epsi_range)
        for i,e in enumerate(epsi_range):
            
            if Type == 'FV':
                
                model.FV_fit(epsi = e)
                
            if Type == 'GE':
                
                model.GE_fit(epsi = e)
            
            error[i] = np.linalg.norm(model.dy_org - model(model.X_org)[1])/np.linalg.norm(model.dy_org)
  
        opt_epsi = epsi_range[np.argmin(error)]
        #train the model with the optimin shape parameter
        if Type == 'FV':
            
            model.FV_fit(epsi = opt_epsi)
            
        if Type == 'GE':
            
            model.GE_fit(epsi = opt_epsi)
        
        #plot error vs shape parameter
        if fig:
            plt.figure()
            plt.loglog(epsi_range[::-1], error[::-1])
            plt.xlabel('Shape Parameter')
            plt.ylabel('Relative Error')
            plt.title('Log of Error vs Shape Parameter')
            plt.loglog(opt_epsi, np.min(error), 'k.')
            
        return opt_epsi
        

class LinearTransformation:
    '''
    Linear transformations to create an isotropic reference frame for model construction.
    '''
    
    @staticmethod
    def ASM(model):
        """
        Complete the active subspace method.

        Parameters:
        model : rbf_model instance
            instance of the rbf model.

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
    
    @staticmethod
    def GE_LHM(model, n_neighbors=None):
        """
        Computes a Symmetric Rank-One (SR1) Hessian approximation for each point in X
        using the n_neighbors+1 closest points.

        Parameters:
        model : rbf_model instance
            instance of the rbf model.
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

        return H_mean
    
    @staticmethod
    def FV_LHM(model):
        """
        Computes function quadratic fit to complete a Hessian approximation for each point in X
        using the n_neighbors+1 closest points.

        Parameters:
        model : rbf_model instance
            instance of the rbf model.

        Returns:
        h_mean : np.ndarray
            The approximation of the average local Hessian.
            Also adds the eigenvalues and vectors as attributes to the model object.
        """

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
    
    @staticmethod
    def Transform(model, method='GE-LHM', n_neighbors=None, 
                                          R=None, S=None):
        """
        Transforms the model according to a chosen method.

        Parameters:
        model : rbf_model instance
            instance of the rbf model.
        method : string, optional
            Which transformation method to use. Default is 'GE-LHM'.
        n_neighbors : int, optional
            Neighbours for LHM methods. Default is None, set to n+1
        R : numpy array, optional
            User-defined rotation matrix. Default is None.
        S : numpy array, optional
            User-defined scalers. Default is None.

        Returns:
        None.
        """

        if method == 'GE-LHM':
            LinearTransformation.GE_LHM(model, n_neighbors=n_neighbors)
        
        if method == 'GE-DLHM':
            H = LinearTransformation.GE_LHM(model, n_neighbors=n_neighbors)

            model.eig, model.evec = np.diag(H), np.identity(model._n_features)

        if method == 'FV-LHM':
            LinearTransformation.FV_LHM(model)

        if method == 'ASM':
            LinearTransformation.ASM(model)

        if method == 'ideal':
            model.evec = R
            model.eig = S
            
        model.scalers = model.eig**0.5 
 
        model.X_org = np.copy(model.X)
        model.X = model.X @ model.evec * model.scalers

        model.C_org = np.copy(model.C)
        model.C = model.C @ model.evec * model.scalers

        if model.dy is not None:
            model.dy_org = np.copy(model.dy)
            model.dy = model.dy @ model.evec / model.scalers

        return