# GE RBF

[Code Docs](Code%20Docs%20867e4dd80a0c4567bff85a8b6d2907cd.csv)

[Documentation](https://www.notion.so/Documentation-ace68552f8e64f459334c0854cd471fb?pvs=21)

# Documentation

> 📘 The main purpose of this code is to use sampled gradient information to construct radial basis function models. Typically, these models are only trained with function value information, instead, in this implementation the gradient information is used in fitting the model, hyper-parameter selection, and preprocessing the data.
> 

## Model Details

### The Kernel

The guassain kernel is the chosen implemented kernel, where $\epsilon$ or`epsi` is the shape hype-parameter. 

<aside>

$$
\phi(\mathbf{x}, \mathbf{c}, \epsilon) = e^{\epsilon||\mathbf{x} - \mathbf{c}||^2}
$$

</aside>

### The Kernel Jacobian

To fit to sampled gradient information, the gradient of the guassain kernel with respect to the input variables `X` is required. This also allows for the model to predict gradient information.

<aside>

$$
\frac{d\phi(\mathbf{x}, \mathbf{c}, \epsilon)}{d\mathbf{x}} = -2 \epsilon (\mathbf{x} - \mathbf{c}) e^{\epsilon||\mathbf{x} - \mathbf{c}||^2}
$$

</aside>

## Model Capabilities

| Model | Fit to Function Values | Fit to gradient vectors |
| --- | --- | --- |
| FV - RBF | ✅ | ❌ |
| GE - RBF | ✅ | ✅ |
| GO - RBF | ❌ | ✅ |

| Hyper-parameter Selection Strategies | Function Value Based | Gradient Information Based |
| --- | --- | --- |
| Cross validation Kfold | ✅ | ✅ |
| Validation Set | ✅ | ❌ |
| Gradient Validation | ❌ | ✅ |

| Preprocessing | Uncoupled Rotation | Isotropic Scaling |
| --- | --- | --- |
| ASM | ✅ | ❌ |
| GE LHM | ✅ | ✅ |
| FV LHM | ✅ | ✅ |

## Model Initialization

Initialize the model with the sampled data.

**Input Parameters**

| Property | Type | Description | Default |
| --- | --- | --- | --- |
| `X` | `numpy`Matrix   | Locations of sampled information. | - |
| `y` | `numpy`vector | Sampled function value information. | - |
| `dy` | `numpy` Matrix | Sampled gradient information. | `None` |
| `centres` | `numpy` Matrix | Locations of the kernel functions. | `None` ; Set to `X` |

## Model Fitting

Find the optimum coefficients for the different model types.

### Function value fit

**Input Parameters**

| Property | Type | Description | Default |
| --- | --- | --- | --- |
| `self` | `RBFmodel`  | Instance of the rbf model. | - |
| `epsi` | `float` scalar | Shape parameter for the gaussian kernels | 1 |

```python
def FV_fit(self, epsi = 1):

	return
```

### Gradient Enhanced Fit

**Input Parameters**

| Property | Type | Description | Default |
| --- | --- | --- | --- |
| `self` | `RBFmodel`  | Instance of the rbf model. | - |
| `epsi` | `float` scalar | Shape parameter for the gaussian kernels | 1 |

```python
def GE_fit(self, epsi = 1):

	return
```

### Gradient Only Fit

**Input Parameters**

| Property | Type | Description | Default |
| --- | --- | --- | --- |
| `self` | `RBFmodel`  | Instance of the rbf model. | - |
| `epsi` | `float` scalar | Shape parameter for the gaussian kernels | 1 |

```python
def GO_fit(self, epsi = 1):

	return
```

## Model Prediction

Predict function and gradient information at new data points

**Input Parameters**

| Property | Type | Description | Default |
| --- | --- | --- | --- |
| `self` | `RBFmodel`  | Instance of the RBF model. | - |
| `Xnew` | `numpy` Matrix | Locations where the model needs to make predictions | - |
| `OnlyFunc` | `boolean` | Specify in the model should only predict function information | `False` |

```python
def __call__(self, Xnew, OnlyFunc = False):

	return
```

Table of Contents

---