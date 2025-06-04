# Hierarchical Bayesian Approaches to Shrinkage and Sparse Estimation

This repository contains MATLAB code that demonstrates hierarchical priors for shrinkage and variable selection, following the comprehensive treatment in Korobilis and Shimizu (2022) "Bayesian Approaches to Shrinkage and Sparse Estimation."

## Monograph Information

**Citation**: Korobilis, D. and Shimizu, K. (2022). "Bayesian Approaches to Shrinkage and Sparse Estimation," *Foundations and Trends in Econometrics*, 11(4), 230-354.

**DOI**: [10.1561/0800000041](http://dx.doi.org/10.1561/0800000041)

**Abstract**: In all areas of human knowledge, datasets are increasing in both size and complexity, creating the need for richer statistical models. This trend is also true for economic data, where high-dimensional and nonlinear/noparametric inference is the norm in several fields
of applied econometric work. The purpose of this paper is to introduce the reader to the realm of Bayesian model determination, by surveying modern shrinkage and variable selection algorithms and methodologies. Bayesian inference is a natural probabilistic framework for quantifying uncertainty and learning about model parameters, and this feature is particularly important for inference in modern models of high dimensions and increased complexity.

We begin with a linear regression setting in order to introduce various classes of priors that lead to shrinkage/sparse estimators of comparable value to popular penalized likelihood estimators (e.g. ridge, lasso). We explore various methods of exact and approximate inference, and discuss their pros and cons. Finally, we explore how priors developed for the simple regression setting can be extended in a straightforward way to various classes of interesting econometric models. In particular, the following case-studies are considered, that demonstrate application of Bayesian shrinkage and variable selection strategies to popular econometric contexts: 
i) vector autoregressive models; ii) factor models; iii) time-varying parameter regressions; iv) confounder selection in treatment effects
models; and v) quantile regression models.
A MATLAB package and an accompanying technical manual allow the reader to replicate many of the algorithms described in this review.


## Supporting Documents

📄 **Working Paper Version**: [Download PDF](https://www.dropbox.com/s/x5uobzgj16imar2/2021.11.25_BMD.pdf?dl=0)

📄 **Technical Appendix**: [Download PDF](https://www.dropbox.com/s/nn87tulmp4xazqe/2021.11.24_Techincal_Document.pdf?dl=0)  
*Complete technical details including posterior conditionals for MCMC algorithms*

📄 **Code Manual**: [Download PDF](https://www.dropbox.com/s/hsp9rdktrih1cqr/Manual.pdf?dl=0)  
*Detailed documentation for the accompanying MATLAB code*

## Repository Contents

This repository implements the theoretical concepts from the monograph through practical MATLAB demonstrations covering:

### 1. Foundational Methods
- **Linear regression with shrinkage priors**
- **Comparison with penalized likelihood estimators** (Ridge, LASSO)
- **Various methods of exact and approximate inference**
- **Hierarchical prior specifications**

### 2. Econometric Applications

The code demonstrates how shrinkage and variable selection priors extend to important econometric contexts:

#### A. Vector Autoregressive (VAR) Models
- Bayesian VAR with shrinkage priors
- Variable selection in high-dimensional VARs
- Minnesota-type priors and extensions

#### B. Factor Models
- Sparse factor models
- Variable selection for factor loadings
- Hierarchical priors for factor structures

#### C. Time-Varying Parameter Regressions
- TVP models with shrinkage
- Variable selection over time
- Adaptive hierarchical priors

#### D. Treatment Effects Models
- Confounder selection in causal inference
- Sparse propensity score models
- High-dimensional treatment effect estimation

#### E. Quantile Regression Models
- Bayesian quantile regression
- Variable selection across quantiles
- Hierarchical priors for quantile-specific effects

## Key Methodological Features

### Shrinkage Priors Implemented
- **Ridge-type (Gaussian) priors**
- **LASSO-type (Laplace) priors**
- **Spike-and-slab priors**
- **Horseshoe priors**
- **Dirichlet-Laplace priors**
- **Global-local shrinkage hierarchies**

### Inference Methods
- **Markov Chain Monte Carlo (MCMC)**
- **Variational Bayes approximations**
- **Expectation-Maximization (EM) algorithms**
- **Gibbs sampling implementations**

### Model Selection Approaches
- **Automatic variable selection**
- **Adaptive shrinkage parameters**
- **Hierarchical model specifications**
- **Empirical Bayes methods**

## Quick Start Guide

### Requirements
- MATLAB R2016b or later
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox (for some advanced features)

### Installation
```bash
git clone https://github.com/korobilis/hierarchicalbayes.git
cd hierarchicalbayes
```

### Basic Usage
```matlab
% Add repository to MATLAB path
addpath(genpath('.'))

% Run basic linear regression example
run_linear_regression_example.m

% Explore VAR with shrinkage priors
run_var_shrinkage_example.m

% Try factor model with variable selection
run_factor_model_example.m
```

## Code Organization

```
hierarchicalbayes/
├── README.md                                    # Main repository overview
├── Bayesian_Model_Determination.pdf            # Bayesian model determination strategies
├── Manual.pdf                                  # Comprehensive user manual
├── Techincal_Document.pdf                      # Technical methodologies and foundations
├── BayesHDTE/                                  # Bayesian High-Dimensional Treatment Effects
├── Causal Inference/                           # Confounder selection and causal models
├── Linear Regression/                          # Linear regression with shrinkage priors
├── Synthetic experiments/                      # Simulation studies and comparisons
│   └── cond_vs_uncond/                        # Conditional vs unconditional model comparisons
├── Time-varying parameter models/              # TVP model implementations
└── VARs/                                       # Vector Autoregressive models with hierarchical priors
```

## Theoretical Background

### Bayesian Shrinkage Philosophy

The code implements the core principle that **hierarchical Bayesian methods** provide:

1. **Automatic regularization** through prior specifications
2. **Uncertainty quantification** via posterior distributions  
3. **Adaptive shrinkage** that learns from data
4. **Model selection** integrated with parameter estimation

### Global-Local Shrinkage Hierarchy

Many implementations follow the **global-local paradigm**:

```
β_j | τ_j, λ ~ N(0, τ_j² λ²)    [Local shrinkage]
τ_j ~ π(τ_j)                     [Local scale parameters]  
λ ~ π(λ)                         [Global shrinkage]
```

### Computational Advantages

- **Conjugate priors** enable efficient Gibbs sampling
- **Variational approximations** for large-scale problems
- **Adaptive algorithms** that tune hyperparameters automatically

## Data and Examples

### Simulated Data
- **High-dimensional regression** settings
- **Sparse coefficient structures**
- **Time-varying parameter scenarios**
- **Treatment effect simulations**

### Empirical Applications
- **Macroeconomic forecasting** with many predictors
- **Financial variable selection**
- **Cross-sectional prediction** problems
- **Panel data applications**

## Comparison with Classical Methods

The code systematically compares Bayesian approaches with:

- **Penalized likelihood methods** (Ridge, LASSO, Elastic Net)
- **Information criteria** (AIC, BIC)
- **Cross-validation approaches**
- **Classical variable selection procedures**

## Advanced Features

### Computational Efficiency
- **Vectorized MATLAB implementations**
- **Memory-efficient algorithms**
- **Parallel computing support** (where applicable)
- **Convergence acceleration techniques**

### Robustness Checks
- **Multiple chain diagnostics**
- **Sensitivity analysis tools**
- **Prior specification robustness**
- **Out-of-sample validation**

## Educational Value

This repository serves as:

- **Practical implementation** of theoretical concepts
- **Benchmark comparison** between methods
- **Template code** for extending to new applications
- **Learning resource** for Bayesian econometrics

## Important Notes

⚠️ **Prerequisites**: A solid understanding of Bayesian econometrics is assumed, including:
- Prior distributions and their properties
- MCMC methods and convergence diagnostics  
- Model comparison and selection criteria

⚠️ **Computational Requirements**: Some applications (especially high-dimensional VARs) are computationally intensive and may require substantial computing resources.

⚠️ **No Technical Support**: This code is provided for educational and research purposes without ongoing technical support.

## Citation Guidelines

**Primary Citation** (Always required):
```
Korobilis, D. and Shimizu, K. (2022). Bayesian Approaches to Shrinkage and Sparse Estimation. 
Foundations and Trends in Econometrics, 11(4), 230-354.
```

**Working Paper Version** (Alternative):
```
Korobilis, D. and Shimizu, K. (2021). Bayesian Approaches to Shrinkage and Sparse Estimation. 
arXiv:2112.11751.
```

## Related Resources

### Additional Reading
- **Technical Appendix**: Complete derivations and implementation details
- **Code Manual**: Step-by-step usage instructions
- **Original monograph**: Theoretical foundation and literature review

### Extensions and Applications
- Vector autoregressive forecasting applications
- Factor model implementations  
- Time-varying parameter extensions
- Treatment effect methodology

## License

This code is provided for academic and research purposes. Please cite the original monograph when using or referring to this code.

## Authors

**Dimitris Korobilis**  
University of Glasgow  
Email: dimitris.korobilis@glasgow.ac.uk

**Kenichi Shimizu**  
University of Glasgow 

## Disclaimer

This code is designed for researchers and advanced students with experience in Bayesian econometrics. The algorithms are research-grade implementations intended for academic use. Users should validate results and ensure appropriateness for their specific applications.

---

*Repository last updated: 04 June 2025*  
*For questions about methodology, please refer to the original monograph and technical appendix.*
