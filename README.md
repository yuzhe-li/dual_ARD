# Dual automatic relevance determination for linear latent variable models and its application to calcium imaging data analysis


This repository introduces a **dual automatic relevance determination (dual ARD)** method designed for linear latent variable models. The method has been specifically applied to calcium imaging data analysis, demonstrating its utility in effective and automatic dimensionality reduction. 
 



## Requirements
To install dependencies:
```
conda env create -f environment.yml
```
## Usage Examples

An example notebook (`example.ipynb`) is included to demonstrate basic usage of the `Model` class.

### **Run a single model**
```python
from fun_models_class import Model

# Define model
model_name = 'dual_ard'
model = Model(name=model_name)

# Fit the model
model.fit(y)

# Visualize results
model.imshow_wxyq()
```




### Run multiple models for comparison 
```python
from fun_models_class import Models

# Define multiple models
model_names = ['pca', 'bpca_common', 'dual_ard_individual']
models = Models(model_names=model_names)

# Fit models and visualize results
models.fit(y,imshow = True)

# Plot model comparison metrics
models.plot_score_comparsion(score_name)
```




### Method Description:

#### **Class: `Model`**  
**Attributes:**  
- `name`: Specifies the name of the model.



**Available model names:**  
- `'pca'`: Principal Component Analysis.  
- `'ppca'`: Probabilistic PCA.  
- `'fa'`: Factor Analysis.  
- `'bpca_common'`: Bayesian PCA (common variant).  
- `'bpca_individual'`: Bayesian PCA (individual variant).  
- `'dual_ard_common'`: Dual Automatic Relevance Determination (common variant).  
- `'dual_ard_individual'`: Dual ARD (individual variant).  



#### **Functions:**  

##### **`fit`**  
**Arguments:**  
- `y`: Input data to be modeled.  
- `D`: Dimensionality of the input data `y`.  
- `q`: Target dimensionality for dimensionality reduction.  
- `seed`: Random seed for reproducibility of results.  
- `returnQ`: Boolean indicating whether to return detailed model parameters from the variational Bayesian inference process.  
- `verbose`: Specifies the verbosity level for details in the variational Bayesian process (utilizes `bayepy`).  



##### **`imshow_wxyq`**  
Provides a visualization of weights and transformed dimensions.



##### **`plot_score_comparison`**  
Compares performance metrics across different configurations or settings. 
