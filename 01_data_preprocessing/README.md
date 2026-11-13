# Module 01: Data Preprocessing Toolkit

## Mathematical Foundations & Principles

Data preprocessing is the cornerstone of robust machine learning systems. Real-world features often reside on vastly divergent dimensional scales, contain missing entries, and introduce skew that obstructs gradient-based optimization.

### 1. Missing Value Imputation
Given a data matrix $X \in \mathbb{R}^{n \times d}$, let $\Omega_j \subset \{1, \dots, n\}$ be the index set of observed values for feature $j$:

* **Mean Imputation:**
  $$\hat{x}_j = \frac{1}{|\Omega_j|} \sum_{i \in \Omega_j} X_{ij}$$
* **Median Imputation (robust to heavy outliers):**
  $$\hat{x}_j = \text{median}(\{X_{ij} : i \in \Omega_j\})$$
* **KNN Imputation:** For sample $i$ with missing feature $j$, we identify the $k$ nearest neighbors based on observed dimensions using nan-Euclidean distance:
  $$d(u, v) = \sqrt{\frac{d_{\text{total}}}{d_{\text{common}}} \sum_{m \in \text{common}} (u_m - v_m)^2}$$
  $$\hat{X}_{ij} = \frac{1}{k} \sum_{p \in \mathcal{N}_k(i)} X_{pj}$$

### 2. Feature Scaling & Condition Numbers
For gradient descent, if feature scales vary wildly (e.g., age $\in [20, 60]$ vs. income $\in [20000, 150000]$), the loss surface forms narrow, highly eccentric elliptical contours. The condition number $\kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$ of the Hessian matrix becomes enormous, causing severe oscillations.

* **Standard (Z-Score) Scaling:**
  $$z = \frac{x - \mu}{\sigma}, \quad \mu = \frac{1}{n}\sum_{i=1}^n x_i, \quad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2}$$
* **Min-Max Scaling:**
  $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (\text{max}_{\text{target}} - \text{min}_{\text{target}}) + \text{min}_{\text{target}}$$

### 3. Preventing Data Leakage
> [!IMPORTANT]
> A critical failure mode in production ML pipelines is fitting scalers or imputers on the entire dataset prior to splitting. This allows statistical properties of the test set ($\mu_{\text{test}}, \sigma_{\text{test}}$) to contaminate the training set. Always call `.fit()` exclusively on the training partition, and only `.transform()` on validation and testing partitions.

---

## Code Directory

| File | Description |
| :--- | :--- |
| `scratch_preprocessing.py` | Full NumPy implementations of `MissingValueHandlerScratch`, `FeatureScalerScratch`, `CategoricalEncoderScratch`, and stratified `train_test_split_scratch`. |
| `library_preprocessing.py` | Production pipeline with scikit-learn `SimpleImputer`, `KNNImputer`, `StandardScaler`, and `ColumnTransformer`. |
| `compare_preprocessing.py` | Verifies mathematical numerical parity ($<10^{-12}$) and plots gradient descent trajectories under scaled vs unscaled features. |

## Quick Execution

```bash
python3 scratch_preprocessing.py
python3 library_preprocessing.py
python3 compare_preprocessing.py
```
