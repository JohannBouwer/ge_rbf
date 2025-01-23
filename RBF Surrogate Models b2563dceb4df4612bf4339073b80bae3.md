# RBF Surrogate Models

[Code Docs](Code%20Docs%20867e4dd80a0c4567bff85a8b6d2907cd.csv)

[Documentation](https://www.notion.so/Documentation-ace68552f8e64f459334c0854cd471fb?pvs=21)

# Radial Basis Functions

Shape parameter selection using sampled gradient information.

> 📘 The goal is too use the gradient information to improve the selection of hyper-parameters.
> 
> - Implement a iteration loop where the shape parameters are..
> - Use the gradient information to find the error associated with the shape parameter in each direction.

## Capabilities

| Model | Fit to Function Values | Fit to gradient vectors |
| --- | --- | --- |
| FV - RBF | ✅ | ❌ |
| GE - RBF | ✅ | ✅ |
| GO - RBF | ❌ | ✅ |

| Pre-processing | Uncoupled Rotation | Isotropic Scaling |
| --- | --- | --- |
| ASM | ✅ | ❌ |
| LHM | ✅ | ✅ |

| Shape Selection Strategies | Direction Interdependence | Kernel Independence |
| --- | --- | --- |
| tbd | ✅ | ❌ |
| tbd | ❌ | ✅ |
| tbd | ✅ | ✅ |

## The Constructor

Set the model parameters:   `X` `y` `dy` `centres` 

[The Constructor](https://www.notion.so/The-Constructor-9a0afb14bd4245129b896d3815ce64bb?pvs=21)

### Parameters

| Property | Type | Description | Default |
| --- | --- | --- | --- |
| `X` | `numpy` Matrix | Locations of sampled information. | - |
| `y` | `numpy` vector | Sampled function value information. | - |
| `dy` | `numpy` Matrix | Sampled gradient information. | `None` |
| `centres` | `numpy` Matrix | Locations of the kernel functions. | `None` ; Set to `X` |

## The Kernel

We want to input a vector of `epsi` values so that:

1. Each Kernel has its own shape parameter
2. Each direction has its own shape parameter.
3. Both are independent.

<aside>

$$
\phi(\mathbf{x}, \mathbf{c}, \epsilon) = e^{\epsilon||\mathbf{x} - \mathbf{c}||^2}
$$

</aside>

[The Kernel Method](https://www.notion.so/The-Kernel-Method-63250203e1f5441987e9d0e01ee51b3a?pvs=21)

## The Kernel Jacobian

We want the gradient of the output variable `y` with respect to the in the input variables `X`. 

<aside>

$$
\phi(\mathbf{x}, \mathbf{c}, \epsilon) = e^{\epsilon||\mathbf{x} - \mathbf{c}||^2}
$$

</aside>

[The Kernel Jacobian](https://www.notion.so/The-Kernel-Jacobian-28921d72472c4b0682563411e11ab3c1?pvs=21)

## Fitting the Model

Find the optimum coefficients of the model

[Fit the Model](https://www.notion.so/Fit-the-Model-c7bdb8e0c89c427db1f43ae67eec1c03?pvs=21)

## Predict Function Values

Predict values at new data points

[Predict](https://www.notion.so/Predict-ab2bdd90be6544939564ebe5161a09b4?pvs=21)

## Predict Gradients

[Predict Gradients](https://www.notion.so/Predict-Gradients-f546ca4f725345e5b11e7d1bb1f01fda?pvs=21)

## Shape Parameter Selection

[Shape Parameter Selection](https://www.notion.so/Shape-Parameter-Selection-df91b14f0ef2450ead3610129095b732?pvs=21)

## Quick Copies

- **Variable types**
    - `None`
    - `string`
    - `int`
    - `signed`
    - `unsigned`
    - `float`
    - `boolean`
    - `true`
    - `false`
    - `list`
    - `array`
    - `tuple`
    - `range`
    - `dict`
    - `complex`
    - `bytes`
    - `set`
- **Pie chart**
    
    ```mermaid
    %%{init: {'theme': 'default'} }%%
    pie title Common Error Codes
    	"301" : 30
    	"404" : 40
    	"503" : 30
    ```
    
- **Flow chart**
    
    ```mermaid
    flowchart LR
    	I[(Idea)] ---> C(Code)
    	C ---> R(Review)
    	R --> I
    ```
    
- **Mathematical Equations**
    
    *Inline Equation:* $F=G\frac{m_1m_2}{d^2}$
    
    *Inline Equation:* $F=G\frac{m_1m_2}{d^2}$
    
    *Block equation:*
    
    $$
    \frac{df}{dt}=\lim_{h\to 0} \frac{f(t+h)-f(t)}{h}
    $$
    

Table of Contents

---