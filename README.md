# GE_RBF
Radial Basis Functions (RBF) surrogate modeling strategies developed and implemented during my PhD research ([1](https://doi.org/10.1080/15397734.2021.1950549), [2]( https://doi.org/10.3390/mca28020057), [3](https://doi.org/10.1016/j.cma.2023.116648)). This research focused on leveraging gradient information for constructing RBF models and preprocessing sampled data to improve model performance and accuracy.

## Implemented Models
The code includes three types of RBF models in the rbfmodels.py file:

### 1. FV RBF (Function Value RBF Models)
Standard RBF models fitted solely to function values of the underlying function.

### 2. GE RBF (Gradient Enhanced RBF Models)
RBF models that incorporate both sampled function values and gradient information, typically employing a regression-based fitting approach.

### 3. GO RBF (Gradient Only RBF Models)
RBF models fitted exclusively to gradient information.

## Shape-Parameter Selection Strategies
Three strategies for selecting the shape parameter are implemented in the preprocessing.py file:

### 1. K-Fold Cross-Validation Selection
Splits the data into folds to optimize the shape parameter based on cross-validation performance.

### 2. Validation Set Error Selection
Uses a separate validation set to select the shape parameter that minimizes the error.

### 3. Gradient-Based Error Selection
Optimizes the shape parameter based on error estimates derived from gradient information.

## Key Contributions
The primary contribution of this work is the development and implementation of strategies to transform the reference frame of the sampled data into one where the function response is isotropic. This transformation mitigates the limitations of RBF models that use isotropic basis functions or kernels. These methods are also in the preprocessing.py file.

The code includes implementations of the following methods for reference frame transformation:

### 1. Active Subspace Method (ASM)
Identifies dominant directions in the input space where the function exhibits the most variation.

### 2. Gradient-Based Local Hessian Method (GE LHM)
Constructs the transformation using gradient information to approximate local curvature.

### 3. Function-Based Local Hessian Method (FV LHM)
Uses function values to estimate local Hessians and define the transformation.
