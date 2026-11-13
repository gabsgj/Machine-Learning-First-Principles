# Machine Learning Comprehensive Theoretical & Conceptual Reference

> **Target Audience:** Technical interview candidates, machine learning engineers, and researchers.  
> **Scope:** Foundational theory, mathematical derivations, algorithmic lifecycles, performance metrics, and experiment-specific analyses across all 20 curriculum experiments.

---

## Table of Contents

1. [Module 1: Machine Learning Fundamentals & Core Concepts](#module-1-machine-learning-fundamentals--core-concepts)
2. [Module 2: Data Preprocessing, Cleaning & Feature Engineering](#module-2-data-preprocessing-cleaning--feature-engineering)
3. [Module 3: Linear & Polynomial Regression, Regularization & Optimization](#module-3-linear--polynomial-regression-regularization--optimization)
4. [Module 4: Logistic Regression, Classification & Parameter Estimation (MLE / MAP)](#module-4-logistic-regression-classification--parameter-estimation-mle--map)
5. [Module 5: Naïve Bayes Classifiers & Natural Language Processing](#module-5-naïve-bayes-classifiers--natural-language-processing)
6. [Module 6: K-Nearest Neighbors (KNN)](#module-6-k-nearest-neighbors-knn)
7. [Module 7: Decision Trees & ID3 / CART Algorithms](#module-7-decision-trees--id3--cart-algorithms)
8. [Module 8: Support Vector Machines (SVM) & Kernel Methods](#module-8-support-vector-machines-svm--kernel-methods)
9. [Module 9: Artificial Neural Networks & Multilayer Perceptrons (MLP)](#module-9-artificial-neural-networks--multilayer-perceptrons-mlp)
10. [Module 10: Unsupervised Learning & Clustering Algorithms](#module-10-unsupervised-learning--clustering-algorithms)
11. [Module 11: Ensemble Learning & Resampling Methods](#module-11-ensemble-learning--resampling-methods)
12. [Module 12: Model Evaluation, Metrics & Diagnostic Curves](#module-12-model-evaluation-metrics--diagnostic-curves)
13. [Module 13: Experiment-by-Experiment Syllabus Defense (Experiments 1–20)](#module-13-experiment-by-experiment-syllabus-defense-experiments-120)
14. [Module 14: Python, NumPy, Pandas & Scikit-Learn Framework Essentials](#module-14-python-numpy-pandas--scikit-learn-framework-essentials)

---

## Module 1: Machine Learning Fundamentals & Core Concepts

### 1.1 What is Machine Learning?
Machine Learning (ML) is an engineering and statistical discipline within Artificial Intelligence where computer systems learn representations, extract patterns, and approximate decision functions directly from empirical data without relying on hand-crafted, hard-coded rules.

### 1.2 What is the fundamental difference between traditional programming and Machine Learning?
* **Traditional Programming:** Software engineers manually construct explicit rules and algorithms ($Rules + Data \rightarrow Answers$).
* **Machine Learning:** Statistical algorithms ingest observed data and ground-truth target outputs to induce the governing rules automatically ($Data + Answers \rightarrow Rules$).

```
Traditional Programming:   Data + Rules   ───────────────► Answers
Machine Learning:          Data + Answers ───────────────► Rules (Learned Model)
```

### 1.3 How do Artificial Intelligence, Machine Learning, and Deep Learning relate?
* **Artificial Intelligence (AI):** The broad field encompassing any computational system capable of performing tasks requiring human-like intelligence.
* **Machine Learning (ML):** A foundational subset of AI focused on statistical methods and data-driven function estimation.
* **Deep Learning (DL):** A specialized subset of ML utilizing artificial neural networks with multiple representation-learning layers (deep topologies).

### 1.4 What are the primary paradigms of Machine Learning?
1. **Supervised Learning:** Learning a predictive mapping function $f: X \rightarrow Y$ from labelled pairs $(x_i, y_i)$. Includes regression and classification.
2. **Unsupervised Learning:** Discovering latent clusters, probability density distributions, or manifold projections within unlabelled data $\{x_i\}$.
3. **Semi-Supervised Learning:** Training models using a small volume of labelled data alongside a large volume of unlabelled observations.
4. **Reinforcement Learning:** An agent interacts with a dynamic environment, taking actions $a_t$ from states $s_t$ to maximize a cumulative scalar reward signal $r_t$.

### 1.5 Differentiate between Classification and Regression.
* **Classification:** Predicts a discrete categorical class label $y \in \{C_1, C_2, \dots, C_K\}$. Output space is qualitative.
* **Regression:** Predicts an unbounded continuous real-valued quantity $y \in \mathbb{R}$. Output space is quantitative.

### 1.6 What is the distinction between Parametric and Non-Parametric models?
* **Parametric Models:** Assume a fixed mathematical functional form with a fixed number of parameters that does not change with sample size (e.g., Linear Regression, Logistic Regression, Linear SVM). Fast inference, high inductive bias.
* **Non-Parametric Models:** Do not impose rigid functional constraints; model complexity and effective parameter count scale dynamically with the dataset size (e.g., KNN, Decision Trees, Kernel SVM). Highly flexible, lower inductive bias.

### 1.7 What is Inductive Bias?
Inductive bias is the set of fundamental mathematical assumptions that a learner uses to predict outputs for unseen instances outside the training dataset (e.g., Linear Regression assumes linear feature relationships; Decision Trees assume axis-aligned splits; KNN assumes feature-space proximity implies class affinity).

### 1.8 What does the No Free Lunch Theorem state?
The No Free Lunch (NFL) Theorem states that when averaged over all possible data-generating distributions, no machine learning algorithm is universally superior to any other. Superior performance on a specific problem arises strictly because an algorithm's inductive biases match the underlying data distribution.

### 1.9 Differentiate between Model Parameters and Hyperparameters.
* **Model Parameters:** Learned automatically from training data via optimization (e.g., linear regression weights $w$ and intercept $b$, neural network weight matrices $W^{[l]}$).
* **Hyperparameters:** Configured manually by the engineer prior to training to control the learning process and model capacity (e.g., learning rate $\alpha$, regularization penalty $\lambda$, polynomial degree $d$, neighbor count $K$, tree depth `max_depth`).

### 1.10 What is Data Leakage, and why is it dangerous?
Data leakage occurs when information from outside the training partition (such as validation or test data) inadvertently influences model training or preprocessing. For instance, calculating normalization parameters ($\mu, \sigma$) across the entire dataset prior to splitting leaks test distribution information, producing overly optimistic cross-validation results that fail in production.

### 1.11 Define Overfitting and Underfitting.
* **Overfitting (High Variance):** The model learns noise, outliers, and spurious training set patterns. Symptom: High training accuracy, poor validation/test accuracy.
* **Underfitting (High Bias):** The model is too simplistic to capture the underlying structure. Symptom: Poor performance across both training and testing partitions.

### 1.12 Formulate the Bias-Variance Decomposition mathematically.
For a target $y = f(x) + \epsilon$ with zero-mean noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the expected prediction error of an estimator $\hat{f}(x)$ decomposes into:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\Big(\mathbb{E}[\hat{f}(x)] - f(x)\Big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\Big[\big(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\big)^2\Big]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}$$

* Increasing model complexity decreases Bias but increases Variance.
* Optimal generalization occurs at the complexity level where the sum of $\text{Bias}^2 + \text{Variance}$ is minimized.

### 1.13 Differentiate between Batch, Mini-Batch, and Stochastic Gradient Descent.
* **Batch Gradient Descent:** Computes exact gradients across the entire dataset simultaneously. Smooth, deterministic convergence, but memory-intensive.
* **Stochastic Gradient Descent (SGD):** Updates parameters after evaluating each individual training instance. Fast, memory-efficient, but noisy and oscillatory.
* **Mini-Batch Gradient Descent:** Updates parameters after processing a small batch (e.g., 32, 64, 128 instances). Balances SIMD vectorization efficiency with gradient stability.

### 1.14 What is a Stratified Split?
A stratified split ensures that the relative frequency of each target class in both training and test partitions matches the overall population distribution. This prevents minority classes from being excluded during evaluation on imbalanced datasets.

---

## Module 2: Data Preprocessing, Cleaning & Feature Engineering

### 2.1 What are the three mechanisms of missing data?
1. **Missing Completely at Random (MCAR):** Missingness is entirely independent of observed and unobserved data.
2. **Missing at Random (MAR):** Missingness depends systematically on observed features, but not on the missing value itself.
3. **Missing Not at Random (MNAR):** Missingness is directly related to the unobserved value itself (e.g., high earners refusing to state income).

### 2.2 Compare Dropping vs. Imputing missing values.
* **Dropping (Listwise Deletion):** Removes rows containing null values. Preserves feature distribution without distortion, but discards data volume and risks severe selection bias if data is not MCAR.
* **Imputing:** Fills missing entries using summary statistics (mean, median, mode) or predictive models (KNN, iterative imputers). Retains sample size but may artificially reduce feature variance.

### 2.3 When should the Median be preferred over the Mean for imputation?
Use the **Median** when a numerical feature has a skewed distribution or contains extreme outliers. The mean is pulled by extreme values, whereas the median is an outlier-resistant robust statistic.

### 2.4 Why must preprocessing transformers be fit ONLY on the training split?
To preserve strict statistical independence and prevent data leakage. Summary statistics ($\mu, \sigma, x_{\min}, x_{\max}$) must reflect only historical training information. Fitting on the entire dataset incorporates test distribution knowledge into model inputs.

### 2.5 Contrast Standardization (Z-score) and Min-Max Normalization.
* **Standardization (Z-score):** Centers features to zero mean and unit standard deviation:
  $$z = \frac{x - \mu}{\sigma}$$
  Does not constrain values to a bounded interval; preserves outlier visibility.
* **Min-Max Normalization:** Rescales values linearly into a fixed range, typically $[0, 1]$:
  $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
  Compresses in-range variance if extreme outliers are present.

### 2.6 Why do distance-based algorithms require feature scaling, while Decision Trees do not?
* **Distance-based algorithms (KNN, SVM, K-Means):** Compute geometric distances (e.g., Euclidean metric $\sqrt{\sum (u_j - v_j)^2}$). Features with larger numerical scales dominate distance calculations.
* **Gradient-based algorithms (Logistic Regression, Neural Networks):** Unscaled features warp loss contours into elongated ellipses, causing gradient descent to oscillate.
* **Decision Trees:** Split features based on monotonic threshold inequalities ($x_j \le \theta$). Monotonic linear transformations do not alter split ordering or information gain.

### 2.7 Differentiate between One-Hot Encoding and Label Encoding.
* **Label Encoding:** Maps categories to sequential ordinal integers ($0, 1, 2, \dots$). Suitable only for ordinal features with a natural rank (e.g., Small < Medium < Large). When applied to nominal variables (e.g., Red, Blue), linear models fabricate artificial distances.
* **One-Hot Encoding:** Creates binary indicator columns for each category ($0$ or $1$). Prevents false ordering.

### 2.8 What is the Dummy Variable Trap, and how is it resolved?
When categorical variables with $K$ categories are one-hot encoded, the sum of the $K$ columns equals a vector of ones, perfectly duplicating the model intercept. This introduces exact multicollinearity, making $(X^T X)$ singular.
* **Resolution:** Drop one baseline reference column, retaining $K-1$ indicator features (`drop_first=True`).

### 2.9 How are outliers detected using the Interquartile Range (IQR) rule?
Given the first quartile ($Q_1$, 25th percentile) and third quartile ($Q_3$, 75th percentile):
$$\text{IQR} = Q_3 - Q_1$$
An observation $x$ is flagged as an outlier if:
$$x < Q_1 - 1.5 \times \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \times \text{IQR}$$

### 2.10 Contrast Feature Selection and Feature Extraction.
* **Feature Selection:** Selects a subset of the original input features without altering their representation (e.g., Recursive Feature Elimination, Lasso $L_1$ shrinkage, Information Gain). Retains original interpretability.
* **Feature Extraction:** Projects high-dimensional inputs into a lower-dimensional transformed space (e.g., Principal Component Analysis, Linear Discriminant Analysis). Creates composite features, losing direct interpretability.

---

## Module 3: Linear & Polynomial Regression, Regularization & Optimization

### 3.1 State the mathematical formulation of Multiple Linear Regression.
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = w^T x + b = X \theta$$
Where $X \in \mathbb{R}^{n \times (d+1)}$ includes an initial column of ones for the bias, and $\theta = [w_0, w_1, \dots, w_d]^T$.

### 3.2 State the Ordinary Least Squares (OLS) Cost Function.
$$J(\theta) = \frac{1}{2n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 = \frac{1}{2n} \|X\theta - y\|_2^2$$
The constant $\frac{1}{2}$ cancels out during differentiation: $\frac{\partial}{\partial \theta} \frac{1}{2} (X\theta - y)^2 = X^T(X\theta - y)$.

### 3.3 Derive the Normal Equation for Linear Regression.
1. Expand the Mean Squared Error in matrix form:
   $$J(\theta) = \frac{1}{2n} (X\theta - y)^T (X\theta - y) = \frac{1}{2n} \Big( \theta^T X^T X \theta - 2 y^T X \theta + y^T y \Big)$$
2. Take the gradient with respect to parameter vector $\theta$:
   $$\nabla_\theta J(\theta) = \frac{1}{n} \Big( X^T X \theta - X^T y \Big)$$
3. Set the gradient to zero:
   $$X^T X \theta^* - X^T y = 0 \implies X^T X \theta^* = X^T y$$
4. Invert $(X^T X)$ to obtain the closed-form analytical solution:
   $$\theta^* = (X^T X)^{-1} X^T y$$

### 3.4 What is the computational complexity of the Normal Equation, and when does it fail?
* **Complexity:** Computing $(X^T X)$ takes $O(n d^2)$, and inverting $(X^T X)^{-1}$ takes $O(d^3)$. As feature count $d$ exceeds $10^4$, analytical inversion becomes computationally prohibitive.
* **Failure Modes:** $(X^T X)$ is singular (non-invertible) if:
  1. Perfect multicollinearity exists (one feature is an exact linear combination of another).
  2. The number of features exceeds observations ($d > n$).

### 3.5 State the Gradient Descent parameter update rule for Linear Regression.
$$\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta) = \theta - \frac{\alpha}{n} X^T (X\theta - y)$$
Where $\alpha > 0$ is the learning rate.

### 3.6 Why is Ordinary Least Squares guaranteed to converge to the global minimum with Gradient Descent?
The Mean Squared Error cost function is strictly convex (its Hessian matrix $\nabla^2 J(\theta) = \frac{1}{n} X^T X$ is positive semi-definite). A convex surface has a single stationary point, ensuring that gradient descent reaches the global minimum without getting trapped in local minima.

### 3.7 What are the five classical Gauss-Markov assumptions of Linear Regression?
1. **Linearity in Parameters:** The relationship between predictors and the target expectation is linear.
2. **Strict Exogeneity:** Residual errors have zero conditional mean ($\mathbb{E}[\epsilon | X] = 0$).
3. **No Multicollinearity:** Columns of $X$ are linearly independent; $\text{rank}(X) = d+1$.
4. **Homoscedasticity:** Error terms possess constant finite variance across all input ranges ($\text{Var}(\epsilon_i | X) = \sigma^2$).
5. **No Autocorrelation:** Residuals are mutually uncorrelated ($\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$).

### 3.8 Define and interpret the Coefficient of Determination ($R^2$ Score).
$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
* **Interpretation:** The fraction of variance in the dependent variable explained by the model predictors.
* $R^2 = 1.0$: Perfect fit.
* $R^2 = 0.0$: Model performs no better than the sample mean baseline $\bar{y}$.
* $R^2 < 0.0$: Model performs worse on test data than a flat line through the mean.

### 3.9 Why is Adjusted $R^2$ preferred over standard $R^2$ in multiple regression?
Standard $R^2$ monotonically increases whenever any new feature is added, even if that feature is random noise. **Adjusted $R^2$** penalizes redundant, non-informative variables:
$$R^2_{\text{adj}} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - d - 1} \right]$$
Adjusted $R^2$ increases only if the new feature improves model explanatory power beyond random chance.

### 3.10 Why is Polynomial Regression classified as a "Linear" Model?
Polynomial Regression maps an input $x$ into higher-order powers $\phi(x) = [1, x, x^2, \dots, x^d]^T$. Although the fitted hypothesis is non-linear in the input space, it remains strictly **linear with respect to parameters $\theta$**:
$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d = \theta^T \phi(x)$$
Because the loss gradient remains linear in $\theta$, it can still be solved analytically via the Normal Equation.

### 3.11 Compare Ridge ($L_2$) and Lasso ($L_1$) Regularization.

| Dimension | Ridge Regression ($L_2$) | Lasso Regression ($L_1$) |
| :--- | :--- | :--- |
| **Penalty Term** | $\lambda \sum_{j=1}^d w_j^2$ | $\lambda \sum_{j=1}^d \|w_j\|$ |
| **Objective Function** | $\frac{1}{2n} \|Xw - y\|^2 + \frac{\lambda}{2} \|w\|_2^2$ | $\frac{1}{2n} \|Xw - y\|^2 + \lambda \|w\|_1$ |
| **Weight Effect** | Asymptotically shrinks weights toward zero | Drives uninformative weights strictly to zero |
| **Feature Selection** | No (retains all features with small weights) | Yes (produces sparse weight vectors) |
| **Analytical Form** | $w^* = (X^T X + \lambda I)^{-1} X^T y$ | No closed-form (requires subgradient or coordinate descent) |
| **Multicollinearity** | Stable (averages weights among correlated features) | Arbitrarily selects one feature from a correlated group |

### 3.12 Provide the geometric intuition for why Lasso produces sparse weights while Ridge does not.
* **Ridge ($L_2$):** Constraint region is a smooth, continuous Euclidean sphere ($\sum w_j^2 \le t$). Elliptical OLS loss contours intersect this smooth sphere at tangent points away from coordinate axes, keeping weights non-zero.
* **Lasso ($L_1$):** Constraint region is a diamond-shaped polytope with sharp vertices located on the coordinate axes ($\sum |w_j| \le t$). Expanding elliptical loss contours typically contact these sharp vertices first, forcing one or more parameters directly to zero.

---

## Module 4: Logistic Regression, Classification & Parameter Estimation (MLE / MAP)

### 4.1 Define the Sigmoid (Logistic) activation function and state its derivative.
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$
* Maps real-valued inputs $z \in (-\infty, +\infty)$ into valid class probabilities $\sigma(z) \in (0, 1)$.
* **Derivative:**
  $$\frac{d\sigma(z)}{dz} = \sigma(z) (1 - \sigma(z))$$

### 4.2 What are Odds and the Logit function?
* **Odds:** The ratio of the probability of an event occurring to the probability of it not occurring:
  $$\text{Odds} = \frac{p}{1 - p}$$
* **Logit Function (Log-Odds):** The natural logarithm of the odds, which maps bounded probabilities $(0, 1)$ linearly onto $(-\infty, +\infty)$:
  $$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) = w^T x + b$$

### 4.3 Why is Mean Squared Error unsuitable for training Logistic Regression?
1. **Non-Convexity:** Passing the non-linear sigmoid through squared error produces a non-convex loss surface with multiple local minima and saddle points, causing gradient descent to stall.
2. **Vanishing Gradients:** When the model makes a highly confident error ($z \gg 0$ when $y=0$), the derivative $\sigma'(z) \rightarrow 0$, halting parameter updates when error is highest.

### 4.4 Formulate the Binary Cross-Entropy Loss (Log Loss) function.
$$J(w) = -\frac{1}{n} \sum_{i=1}^n \Big[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \Big]$$
Where $\hat{p}_i = \sigma(w^T x_i + b)$. This loss function is derived from the Bernoulli negative log-likelihood and is strictly convex.

### 4.5 Derive Maximum Likelihood Estimation (MLE) for Logistic Regression.
1. Given independent Bernoulli trials $P(y_i|x_i; w) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$:
   $$L(w) = \prod_{i=1}^n \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$
2. Take the natural logarithm to obtain the log-likelihood:
   $$\ell(w) = \sum_{i=1}^n \Big[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \Big]$$
3. Maximizing log-likelihood is equivalent to minimizing negative log-likelihood:
   $$\min_w J(w) = -\ell(w)$$
   This confirms that **Binary Cross-Entropy is mathematically identical to Bernoulli Negative Log-Likelihood**.

### 4.6 Formulate Maximum A Posteriori (MAP) Estimation.
Using Bayes' Theorem in parameter space:
$$P(w | D) = \frac{P(D | w) P(w)}{P(D)} \propto P(D | w) P(w)$$
$$\hat{w}_{\text{MAP}} = \arg\max_w \Big[ \ln P(D | w) + \ln P(w) \Big] = \arg\min_w \Big[ -\ell(w) - \ln P(w) \Big]$$
Where $P(w)$ is the prior probability distribution placed over the model weights.

### 4.7 Prove the fundamental duality between MAP Estimation and Regularization.
1. **Gaussian Prior ($L_2$ Regularization):**
   Assume a zero-mean isotropic Gaussian prior: $w \sim \mathcal{N}(0, \sigma_0^2 I) \implies P(w) \propto \exp\left(-\frac{\|w\|_2^2}{2\sigma_0^2}\right)$.
   $$-\ln P(w) = \frac{1}{2\sigma_0^2} \|w\|_2^2 + \text{const}$$
   Substituting into the MAP objective:
   $$J_{\text{MAP}}(w) = -\ell(w) + \frac{\lambda}{2} \|w\|_2^2 \quad \text{where } \lambda = \frac{1}{\sigma_0^2}$$
   *Gaussian MAP is equivalent to $L_2$ Ridge Regularization.*

2. **Laplace Prior ($L_1$ Regularization):**
   Assume an independent zero-mean Laplace prior: $P(w) \propto \exp\left(-\frac{\|w\|_1}{b}\right)$.
   $$-\ln P(w) = \frac{1}{b} \|w\|_1 + \text{const}$$
   *Laplace MAP is equivalent to $L_1$ Lasso Regularization.*

### 4.8 How does Maximum Likelihood Estimation fail in Multinomial Count distributions, and how does Dirichlet MAP resolve it?
* **MLE Failure (Zero Frequency Problem):** For vocabulary token $j$ with count $c_j$ across total tokens $N$, MLE estimates:
  $$\hat{\theta}_j^{\text{MLE}} = \frac{c_j}{N}$$
  If an unseen word has $c_j = 0$, $\hat{\theta}_j = 0$. In Naive Bayes, multiplying probabilities zeroes out the entire document posterior: $P(c|d) \propto P(c) \prod 0 = 0$.
* **Dirichlet Conjugate Prior (MAP):** Imposing a symmetric Dirichlet prior $\text{Dir}(\alpha)$ yields the MAP estimate:
  $$\hat{\theta}_j^{\text{MAP}} = \frac{c_j + \alpha_j - 1}{N + \sum_{k=1}^V (\alpha_k - 1)}$$
  Setting $\alpha_j = 2$ yields **Laplace (Add-1) Smoothing**:
  $$\hat{\theta}_j = \frac{c_j + 1}{N + V}$$

---

## Module 5: Naïve Bayes Classifiers & Natural Language Processing

### 5.1 State Bayes' Theorem and define its components.
$$P(y | X) = \frac{P(X | y) P(y)}{P(X)}$$
* $P(y | X)$: **Posterior Probability** of class $y$ given observed feature vector $X$.
* $P(X | y)$: **Likelihood** of observing features $X$ given true class $y$.
* $P(y)$: **Prior Probability** of class $y$ before observing features.
* $P(X)$: **Evidence (Marginal Likelihood)** across all classes: $\sum_{c} P(X | c) P(c)$.

### 5.2 What is the "Naïve" assumption in Naïve Bayes?
Naïve Bayes assumes that all input features $x_1, x_2, \dots, x_d$ are **mutually conditionally independent** given the class label $y$:
$$P(x_1, x_2, \dots, x_d | y) = \prod_{j=1}^d P(x_j | y)$$
This simplifies the joint probability estimation from an intractable exponential complexity $O(K \cdot C^d)$ down to a linear parameter complexity $O(K \cdot d)$.

### 5.3 Why does Naïve Bayes perform well in practice despite the independence assumption being violated?
Classification depends on finding the argmax class ($\arg\max_c P(c|X)$), which requires only accurate rank ordering rather than calibrated probability estimates. As long as the true class retains the highest relative probability, violations of independence do not degrade classification accuracy.

### 5.4 Why must computations in Naïve Bayes be conducted in Log-Space?
Multiplying hundreds of small floating-point probabilities ($\prod_{j=1}^d P(x_j|y) \ll 10^{-300}$) triggers arithmetic underflow, rounding the probability to zero. By taking the logarithm, products convert into stable summations:
$$\log P(y | X) \propto \log P(y) + \sum_{j=1}^d \log P(x_j | y)$$

### 5.5 Compare Gaussian, Multinomial, and Bernoulli Naïve Bayes.

| Property | Gaussian NB | Multinomial NB | Bernoulli NB |
| :--- | :--- | :--- | :--- |
| **Input Feature Domain** | Continuous, real-valued ($x \in \mathbb{R}$) | Discrete count vectors ($x \in \mathbb{N}_0$) | Binary indicators ($x \in \{0, 1\}$) |
| **Likelihood Model** | $\mathcal{N}(\mu_{c,j}, \sigma_{c,j}^2)$ | Multinomial distribution | Multivariate Bernoulli |
| **Typical Domain** | Sensor data, physical features | Text classification (word counts) | Short texts, keyword presence/absence |
| **Document Length** | Invariant | Sensitive to word repetitions | Penalizes absent words explicitly |

---

## Module 6: K-Nearest Neighbors (KNN)

### 6.1 Why is KNN characterized as an "Instance-Based" or "Lazy" Learner?
KNN does not construct an explicit internal model or compute parameter updates during a training phase. It simply stores the training instances in memory. All computational work—including pairwise distance calculation, sorting, and voting—is deferred until query time.

### 6.2 Define the classical distance metrics used in KNN.
Given vectors $u, v \in \mathbb{R}^d$:
1. **Euclidean Distance ($L_2$ norm):**
   $$d_2(u, v) = \sqrt{\sum_{j=1}^d (u_j - v_j)^2}$$
2. **Manhattan Distance ($L_1$ norm / Taxicab):**
   $$d_1(u, v) = \sum_{j=1}^d |u_j - v_j|$$
3. **Minkowski Distance ($L_p$ generalized norm):**
   $$d_p(u, v) = \left( \sum_{j=1}^d |u_j - v_j|^p \right)^{1/p}$$

### 6.3 Analyze the impact of the hyperparameter $K$ on Bias and Variance.
* **Small $K$ (e.g., $K=1$):**
  * Training error is identically 0.
  * Low Bias, High Variance.
  * Highly sensitive to noise and outliers, resulting in complex, jagged decision boundaries (Overfitting).
* **Large $K$ (e.g., $K \rightarrow N$):**
  * High Bias, Low Variance.
  * Overly smooth boundary; always predicts the majority class in the dataset (Underfitting).

### 6.4 Why should $K$ be chosen as an odd number in binary classification?
Selecting an odd value for $K$ prevents ties during majority voting, guaranteeing an unambiguous classification decision.

### 6.5 What is the Curse of Dimensionality, and how does it degrade KNN performance?
As feature dimensionality $d$ increases, the volume of the feature space grows exponentially ($V \propto r^d$), causing the data to become sparse. In high-dimensional space, the distance to the nearest neighbor converges to the distance to the farthest neighbor:
$$\lim_{d \rightarrow \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \rightarrow 0$$
Consequently, the concept of spatial proximity breaks down, and all data points become equidistant.

---

## Module 7: Decision Trees & ID3 / CART Algorithms

### 7.1 Define Shannon Entropy and its role in Decision Tree induction.
Shannon Entropy measures the impurity, uncertainty, or disorder of a collection of examples $S$:
$$H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$
Where $p_i$ is the proportion of examples belonging to class $i$.
* $H(S) = 0$: Completely pure node (all instances belong to a single class).
* $H(S) = 1.0$: Maximum impurity for binary classification (50/50 distribution).

### 7.2 Formulate Information Gain for attribute selection.
Information Gain is the expected reduction in entropy achieved by partitioning dataset $S$ on attribute $A$:
$$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
The ID3 algorithm selects the attribute that maximizes Information Gain at each node.

### 7.3 Contrast Information Gain (ID3) with Gini Impurity (CART).
* **Information Gain:** Based on entropy; involves computing logarithms ($\log_2$), which is computationally demanding on large datasets.
* **Gini Impurity:** Measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution:
  $$\text{Gini}(S) = 1 - \sum_{i=1}^c p_i^2$$
  Computationally faster because it avoids transcendental log calculations.

### 7.4 What is the limitation of Information Gain, and how does Gain Ratio resolve it?
* **Limitation:** Information Gain inherently favors high-cardinality attributes with many distinct values (e.g., `Customer_ID`, `Transaction_Date`). Splitting on `Customer_ID` creates pure leaf nodes with zero entropy, yielding high Information Gain but zero generalization value.
* **Gain Ratio (C4.5):** Normalizes Information Gain by the **Split Information** (the entropy of the partition itself):
  $$\text{Gain Ratio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)} \quad \text{where} \quad \text{SplitInfo}(S, A) = -\sum_{v} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

### 7.5 Differentiate between Pre-Pruning and Post-Pruning.
* **Pre-Pruning (Early Stopping):** Halts tree construction early based on predefined threshold criteria (e.g., `max_depth`, `min_samples_split`, `min_impurity_decrease`). Computationally fast, but may stop before discovering multi-attribute interactions (horizon effect).
* **Post-Pruning (Cost-Complexity Pruning):** Grows the tree to maximum depth, then removes branches that contribute minimally to validation accuracy:
  $$R_\alpha(T) = R(T) + \alpha |T|$$
  Where $R(T)$ is misclassification rate, $|T|$ is number of terminal leaves, and $\alpha$ is the complexity penalty parameter.

### 7.6 Compare Decision Trees and Logistic Regression.

| Characteristic | Decision Tree | Logistic Regression |
| :--- | :--- | :--- |
| **Decision Boundary** | Axis-aligned orthogonal hyperplanes (step-like) | Continuous, smooth linear hyperplane |
| **Feature Scaling** | Invariant (scaling does not alter split thresholds) | Mandatory for gradient convergence and regularization |
| **Non-linear Data** | Natively handles non-linear interactions | Requires polynomial expansion or kernel transformations |
| **Interpretability** | Visual decision rules; intuitive logic | Transparent weight coefficients ($w$) and odds ratios |

---

## Module 8: Support Vector Machines (SVM) & Kernel Methods

### 8.1 What is the primary objective of a Support Vector Machine?
SVM identifies the unique optimal separating hyperplane that maximizes the geometric **Margin** between binary classes, minimizing the structural risk of misclassification.

### 8.2 What are Support Vectors?
Support vectors are the critical data points that lie directly on the marginal bounding hyperplanes ($w^T x_i + b = \pm 1$). These vectors define the hyperplane's orientation and position; removing all other training points leaves the decision boundary unchanged.

```
       Class +1          Margin: 2 / ||w||        Class -1
          ●                     │                    ▲
       ●     ● [SV] ────────────┼──────────► w^T x + b = +1
                ●               │
  ──────────────────────────────┼──────────► w^T x + b = 0  (Separating Hyperplane)
                ▲               │
             ▲     ▲ [SV] ──────┼──────────► w^T x + b = -1
          ▲                     │
```

### 8.3 Derive the geometric margin width $M = \frac{2}{\|w\|}$.
1. Let $x_+$ be a support vector on $w^T x + b = +1$ and $x_-$ on $w^T x + b = -1$.
2. The margin width is the scalar projection of $(x_+ - x_-)$ onto the unit normal vector $\frac{w}{\|w\|}$:
   $$M = \frac{w^T (x_+ - x_-)}{\|w\|}$$
3. Since $w^T x_+ = 1 - b$ and $w^T x_- = -1 - b$:
   $$M = \frac{(1 - b) - (-1 - b)}{\|w\|} = \frac{2}{\|w\|}$$
4. Maximizing $\frac{2}{\|w\|}$ is mathematically equivalent to minimizing $\frac{1}{2} \|w\|^2$.

### 8.4 Explain the role of the Soft-Margin hyperparameter $C$ and slack variables $\xi_i$.
For non-linearly separable data, slack variables $\xi_i \ge 0$ allow bounded margin violations:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to } y_i(w^T x_i + b) \ge 1 - \xi_i$$
* **Large $C$ (Strict Margin):** Heavily penalizes misclassifications. Forces a narrower margin, prioritizing training accuracy over margin width (Low Bias, High Variance, risk of Overfitting).
* **Small $C$ (Soft Margin):** Tolerates margin violations. Yields a wider margin, prioritizing generalization over training accuracy (High Bias, Low Variance, risk of Underfitting).

### 8.5 What is the Kernel Trick?
The Kernel Trick allows linear algorithms to operate in high-dimensional (or infinite-dimensional) feature spaces without explicitly computing the coordinates of points in that space. By **Mercer's Theorem**, any positive semi-definite kernel function $K(x, z)$ computes the dot product of mapped representations directly:
$$K(x, z) = \langle \Phi(x), \Phi(z) \rangle$$
This reduces computational complexity from $O(\text{dim}(\Phi))$ down to $O(d)$.

### 8.6 Define the standard SVM Kernel functions.
1. **Linear Kernel:**
   $$K(x, z) = x^T z$$
2. **Polynomial Kernel:**
   $$K(x, z) = (x^T z + c)^d$$
3. **Radial Basis Function (RBF / Gaussian Kernel):**
   $$K(x, z) = \exp(-\gamma \|x - z\|^2) \quad \text{where } \gamma = \frac{1}{2\sigma^2}$$

### 8.7 What is the impact of the RBF Kernel hyperparameter $\gamma$ (gamma)?
* **High $\gamma$:** The Gaussian bell curve has a narrow radius of influence ($\sigma$ is small). The decision boundary forms tight, complex contours around individual training points (Overfitting).
* **Low $\gamma$:** The Gaussian kernel has a wide radius of influence. The decision boundary becomes smooth and nearly linear (Underfitting).

---

## Module 9: Artificial Neural Networks & Multilayer Perceptrons (MLP)

### 9.1 What was the fundamental limitation of the single-layer Perceptron (Minsky & Papert, 1969)?
A single-layer perceptron can learn only linearly separable decision boundaries. It cannot solve non-linear problems like the XOR logical function, which requires at least one hidden layer with non-linear activations.

### 9.2 State the mathematical equations of Forward Propagation for layer $l$.
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$
Where $W^{[l]}$ is the weight matrix, $b^{[l]}$ is the bias vector, $g^{[l]}$ is the non-linear activation function, and $a^{[0]} = X^T$.

### 9.3 Compare Sigmoid, Tanh, and ReLU activation functions.

| Activation | Mathematical Form | Output Range | Key Advantage | Primary Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Probabilistic interpretation | Vanishing gradients; not zero-centered |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Zero-centered outputs | Vanishing gradients for large $|z|$ |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | Constant gradient ($1.0$ for $z>0$); fast computation | Dying ReLU problem (zero gradient for $z<0$) |

### 9.4 What is the Vanishing Gradient Problem?
During backpropagation, gradients are multiplied backwards through layers using the chain rule:
$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \frac{\partial a^{[L]}}{\partial z^{[L]}} \dots \frac{\partial a^{[1]}}{\partial z^{[1]}} \frac{\partial z^{[1]}}{\partial W^{[1]}}$$
For Sigmoid activations, the maximum derivative is $\sigma'(z) \le 0.25$. Multiplying these fractions across several hidden layers causes the gradient to decay exponentially towards zero, preventing early layers from updating their weights.

### 9.5 How does the Backpropagation algorithm work?
Backpropagation applies the calculus **Chain Rule** to compute the partial derivative of the scalar loss function $\mathcal{L}$ with respect to every weight $W^{[l]}$ and bias $b^{[l]}$ in the network, propagating error vectors $\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}$ backwards from output to input.

### 9.6 Define Epoch, Batch Size, and Iteration.
* **Epoch:** One complete forward and backward training pass over all training examples.
* **Batch Size:** The number of training examples evaluated in a single forward/backward pass before updating parameters.
* **Iteration:** A single parameter update step.
  $$\text{Iterations per Epoch} = \frac{\text{Total Training Samples}}{\text{Batch Size}}$$

### 9.7 How does the Adam optimizer combine Momentum and RMSprop?
Adam (Adaptive Moment Estimation) tracks exponentially decaying moving averages of past gradients ($m_t$, first raw moment) and squared gradients ($v_t$, second uncentered moment):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{Momentum: dampens oscillations})$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{RMSprop: scales learning rate per feature})$$
After bias correction ($\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$), weights update via:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## Module 10: Unsupervised Learning & Clustering Algorithms

### 10.1 State the mathematical objective of K-Means Clustering.
K-Means minimizes the **Within-Cluster Sum of Squares (WCSS / Inertia)** across all $K$ clusters:
$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$
Where $\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$ is the centroid of cluster $C_k$.

### 10.2 Describe Lloyd's 2-Step Iterative Algorithm for K-Means.
1. **Assignment Step:** Assign each data point $x_i$ to its nearest centroid based on Euclidean distance:
   $$C_k^{(t)} = \left\{ x_i : \|x_i - \mu_k^{(t)}\|^2 \le \|x_i - \mu_j^{(t)}\|^2 \quad \forall j \right\}$$
2. **Update Step:** Recalculate each centroid as the arithmetic mean of all points assigned to that cluster:
   $$\mu_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{x_i \in C_k^{(t)}} x_i$$
3. Repeat until centroid positions stabilize ($\mu^{(t+1)} \approx \mu^{(t)}$) or cluster assignments cease to change.

### 10.3 How does K-Means++ solve initialization sensitivity?
Standard K-Means randomly picks initial centroids, which often leads to poor local minima. **K-Means++** uses probabilistic seeding:
1. Select the first centroid $\mu_1$ uniformly at random from the data points.
2. For each point $x$, compute its squared distance $D(x)^2$ to the nearest existing centroid.
3. Choose the next centroid with probability proportional to distance:
   $$P(x) = \frac{D(x)^2}{\sum_{i} D(x_i)^2}$$
4. Repeat until all $K$ centroids are chosen. This spaces out initial centroids, improving convergence speed and clustering quality.

### 10.4 How is the optimal number of clusters $K$ determined using the Elbow Method and Silhouette Analysis?
* **Elbow Method:** Plot Inertia (WCSS) versus $K$. As $K$ increases, Inertia decreases monotonically. Look for the "elbow point" where the rate of decrease drops sharply—this marks the point of diminishing returns.
* **Silhouette Score:** Evaluates how well-separated clusters are:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  Where $a(i)$ is the mean distance from point $i$ to all other points in its own cluster, and $b(i)$ is the mean distance to points in the nearest neighboring cluster.
  * $s(i) \approx +1$: Well-clustered and clearly separated.
  * $s(i) \approx 0$: On the decision border between two clusters.
  * $s(i) < 0$: Likely assigned to the wrong cluster.

### 10.5 Compare K-Means and Hierarchical Clustering.

| Feature | K-Means | Agglomerative Hierarchical |
| :--- | :--- | :--- |
| **Cluster Topology** | Spherical, convex clusters | Arbitrary shapes (dependent on linkage) |
| **Complexity** | Fast: $O(n \cdot K \cdot d \cdot \text{iterations})$ | Slower: $O(n^2 \log n)$ to $O(n^3)$ |
| **$K$ Requirement** | Must be chosen prior to training | Determined after viewing the dendrogram |
| **Determinism** | Stochastic (depends on initialization seed) | Deterministic for a fixed distance metric and linkage |

### 10.6 Explain the four standard Linkage Criteria in Hierarchical Clustering.
1. **Single Linkage:** Minimum pairwise distance between points in clusters $A$ and $B$:
   $$d(A, B) = \min_{u \in A, v \in B} d(u, v) \quad (\text{prone to chaining noise})$$
2. **Complete Linkage:** Maximum pairwise distance between points:
   $$d(A, B) = \max_{u \in A, v \in B} d(u, v) \quad (\text{produces compact, spherical clusters})$$
3. **Average Linkage:** Mean pairwise distance:
   $$d(A, B) = \frac{1}{|A||B|} \sum_{u \in A} \sum_{v \in B} d(u, v)$$
4. **Ward's Linkage:** Minimizes the increase in total within-cluster variance upon merging clusters $A$ and $B$:
   $$\Delta \text{ESS}_{AB} = \frac{|A||B|}{|A| + |B|} \|\mu_A - \mu_B\|^2$$

---

## Module 11: Ensemble Learning & Resampling Methods

### 11.1 What is the core philosophy of Ensemble Learning?
Ensemble methods combine predictions from multiple individual models (base estimators / weak learners) to produce a single aggregated prediction that generalizes better and is more robust than any single constituent model.

### 11.2 What is Bagging (Bootstrap Aggregating), and what is its primary mathematical benefit?
Bagging trains multiple independent models in parallel on bootstrap samples drawn uniformly with replacement from the training dataset. Predictions are aggregated via **majority voting** (classification) or **averaging** (regression).
* **Mathematical Benefit:** Substantially reduces **Variance** without increasing Bias. If $M$ independent models each have variance $\sigma^2$ and error correlation $\rho$, the ensemble variance is:
  $$\text{Var}_{\text{ensemble}} = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
  As $M \rightarrow \infty$, ensemble variance drops to $\rho \sigma^2$.

### 11.3 What percentage of unique samples appear in a Bootstrap sample, and what is the Out-of-Bag (OOB) partition?
For a dataset of size $n$, the probability that a specific point is **not** selected in a single draw is $1 - \frac{1}{n}$. After $n$ independent draws with replacement:
$$\lim_{n \rightarrow \infty} \left( 1 - \frac{1}{n} \right)^n = e^{-1} \approx 0.368 \quad (36.8\%)$$
* Approximately **$63.2\%$** of the unique training instances are included in each bootstrap sample.
* The remaining **$36.8\%$** are **Out-of-Bag (OOB)** samples, which serve as an independent validation set to evaluate generalization error without a separate validation split.

### 11.4 How does a Random Forest extend Bagging?
A Random Forest adds a second layer of randomization: **Random Subspace Projection**. At every split in every decision tree, the algorithm evaluates only a random subset of features (typically $m = \sqrt{d}$ for classification or $m = d/3$ for regression). This **decorrelates** the individual trees, lowering the inter-tree correlation $\rho$ and reducing ensemble variance.

### 11.5 What is Boosting, and how does AdaBoost work?
Boosting trains base learners **sequentially**. Each new model prioritizes the training instances that were misclassified by earlier models, primarily reducing **Bias**.
* **AdaBoost Mechanics:**
  1. Initialize equal weights across all samples: $w_i = \frac{1}{n}$.
  2. In round $m$, train a weak learner $h_m(x)$ and compute its weighted error $\epsilon_m$.
  3. Compute the model's voting weight based on accuracy:
     $$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)$$
  4. Increase weights for misclassified samples and decrease weights for correct samples:
     $$w_i \leftarrow w_i \exp\Big( \alpha_m \cdot \mathbb{I}(y_i \neq h_m(x_i)) \Big)$$
  5. Normalize weights and combine models via weighted voting: $H(x) = \text{sign}\left( \sum \alpha_m h_m(x) \right)$.

### 11.6 Compare Bagging and Boosting.

| Dimension | Bagging | Boosting |
| :--- | :--- | :--- |
| **Model Training** | Parallel and independent | Sequential and adaptive |
| **Primary Goal** | Variance reduction (stabilizes noisy models) | Bias reduction (strengthens weak learners) |
| **Base Estimator** | High-capacity, complex models (deep trees) | Low-capacity, weak models (decision stumps) |
| **Overfitting Risk** | Low (adding trees does not cause overfitting) | High if iterations are tuned poorly or data is noisy |

### 11.7 Differentiate between Bootstrapping and K-Fold Cross-Validation.
* **K-Fold Cross-Validation:** Splits data into $K$ equal non-overlapping folds without replacement. Evaluates model stability and performance across unseen partitions; best for model validation.
* **Bootstrapping:** Resamples with replacement from the empirical distribution. Ideal for estimating statistical confidence intervals and standard errors of parameters on small datasets.

---

## Module 12: Model Evaluation, Metrics & Diagnostic Curves

### 12.1 Define the four quadrants of the Confusion Matrix.

| | Actual Positive ($y=1$) | Actual Negative ($y=0$) |
| :--- | :--- | :--- |
| **Predicted Positive ($\hat{y}=1$)** | **True Positive (TP):** Correctly detected positive | **False Positive (FP):** Type I Error (false alarm) |
| **Predicted Negative ($\hat{y}=0$)** | **False Negative (FN):** Type II Error (missed case) | **True Negative (TN):** Correctly rejected negative |

### 12.2 Why does Classification Accuracy fail on imbalanced datasets?
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
If a dataset contains $99\%$ negative samples (e.g., fraud or rare disease detection), a naive baseline model that uniformly predicts negative achieves $99\%$ accuracy while detecting zero true positive cases.

### 12.3 Differentiate between Precision and Recall.
* **Precision:** The proportion of predicted positive cases that are truly positive:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  *Use Case:* When False Positives carry high costs (e.g., spam filtering, content moderation).
* **Recall (Sensitivity / True Positive Rate):** The proportion of actual positive cases that the model successfully detects:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Use Case:* When False Negatives carry high costs (e.g., cancer diagnosis, fraudulent transactions).

### 12.4 Why is the $F_1$-Score calculated as the Harmonic Mean rather than the Arithmetic Mean?
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
The harmonic mean punishes extreme trade-offs between precision and recall. If a model has a Precision of $1.0$ and a Recall of $0.0$, its arithmetic mean is $0.50$, whereas its harmonic mean ($F_1$) is $0.0$, correctly reflecting the model's failure to retrieve positive cases.

### 12.5 What is an ROC Curve, and what does AUC represent?
* **Receiver Operating Characteristic (ROC) Curve:** Plots the True Positive Rate ($\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$) against the False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$) across all possible decision thresholds $\tau \in [0, 1]$.
* **Area Under the Curve (AUC-ROC):** The probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance.
  * $\text{AUC} = 1.0$: Perfect classification.
  * $\text{AUC} = 0.5$: Performance equal to random guessing.

### 12.6 When should a Precision-Recall Curve be chosen over an ROC Curve?
Choose a **Precision-Recall Curve** when the dataset exhibits extreme class imbalance. The ROC curve includes True Negatives in its False Positive Rate denominator ($\frac{\text{FP}}{\text{FP} + \text{TN}}$), which can dilute large spikes in False Positives if True Negatives are overwhelmingly numerous. The PR curve focuses exclusively on the minority positive class.

---

## Module 13: Experiment-by-Experiment Syllabus Defense (Experiments 1–20)

### Experiment 01: Linear Regression & Normal Equation (California Housing)
* **Q: Why use only one feature (MedInc) in Experiment 1?**  
  *A:* To illustrate simple bivariate linear regression, visualize the regression line in 2D, and clearly explain the slope $w$ and intercept $b$.
* **Q: What is the physical interpretation of the slope?**  
  *A:* The expected change in target house value for every one-unit increase in median income.
* **Q: Why standardize before Gradient Descent, and why is it unnecessary for the Normal Equation?**  
  *A:* Gradient descent requires spherical contours to step directly toward the minimum without oscillating. The Normal Equation computes the exact analytical solution in one step, making it invariant to feature scaling.

### Experiment 02: Polynomial Regression (Auto MPG)
* **Q: Why fit polynomial regression on Auto MPG?**  
  *A:* The relationship between horsepower and miles per gallon (MPG) is non-linear—fuel efficiency drops sharply at lower horsepower and flattens at higher horsepower.
* **Q: What happens if the polynomial degree is set too high (e.g., $d=15$)?**  
  *A:* The model overfits, producing severe Runge-phenomenon oscillations between data points that lead to extreme validation errors.

### Experiment 03: Ridge & Lasso Regularization (Diabetes)
* **Q: Why are Ridge and Lasso necessary on the Diabetes dataset?**  
  *A:* The 10 physiological features exhibit multicollinearity. OLS yields inflated parameter variance; regularization stabilizes parameter estimation.
* **Q: How does $\lambda$ (alpha) affect bias and variance?**  
  *A:* Increasing $\lambda$ increases bias and decreases variance by pulling weights toward zero. Setting $\lambda = 0$ reduces both models to Ordinary Least Squares.

### Experiment 04: Logistic Regression with Scaling (Pima Indians / Breast Cancer)
* **Q: What happens to Logistic Regression training if features are not scaled?**  
  *A:* Scikit-learn's default L-BFGS solver fails to converge within its iteration limit (`ConvergenceWarning`). Furthermore, the $L_2$ regularization penalty penalizes unscaled features with small numerical values disproportionately.

### Experiment 05: Parameter Estimation — MLE vs. Dirichlet MAP (Multinomial Distribution)
* **Q: What is the Zero-Frequency problem in NLP?**  
  *A:* If a query term was never observed in a class during training, its MLE probability is $\hat{\theta} = 0$, zeroing out the entire joint document probability.
* **Q: How does the Dirichlet conjugate prior resolve this?**  
  *A:* It acts as pseudo-counts added to the empirical counts. Setting symmetric prior $\alpha=2$ yields Laplace (Add-1) smoothing, ensuring every token retains non-zero probability.

### Experiment 06: Logistic Regression Feature Standardization
* **Q: Why evaluate unscaled vs. scaled Logistic Regression?**  
  *A:* To empirically demonstrate that gradient descent converges substantially faster on standardized inputs, avoiding vanishing weight updates on small-magnitude features.

### Experiment 07: Naïve Bayes Text Classification (20 Newsgroups)
* **Q: Compare Multinomial NB and Bernoulli NB on 20 Newsgroups.**  
  *A:* Multinomial NB tracks word frequency counts, making it ideal for longer documents where key term repetitions carry informative weight. Bernoulli NB tracks binary word presence/absence, performing better on short texts or when word existence alone is informative.

### Experiment 08: Linear SVM & Margin Analysis (Iris Dataset)
* **Q: How are support vectors identified in Scikit-Learn?**  
  *A:* Via the `support_vectors_` attribute, which stores the coordinates of the data points that lie directly on the margins $w^T x + b = \pm 1$.
* **Q: What happens to the margin if $C$ is increased?**  
  *A:* The margin becomes narrower as the penalty for misclassification increases, prioritizing training accuracy over margin width.

### Experiment 09: Decision Tree Induction via Shannon Entropy (Happiness / Income)
* **Q: Why can a Decision Tree achieve 100% training accuracy?**  
  *A:* If no depth constraints (`max_depth=None`) or split restrictions are applied, the tree continues splitting until every leaf node is completely pure ($H(S) = 0$), memorizing the training set.

### Experiment 10: Architectural Comparison — Logistic Regression vs. Decision Tree
* **Q: When does Logistic Regression outperform a Decision Tree?**  
  *A:* When the true underlying decision boundary is approximately linear or planar. Decision trees must approximate smooth diagonal boundaries using axis-aligned orthogonal stair-steps, which requires many splits and risks overfitting.

### Experiment 11: Non-Linear SVM Kernels (Iris / Digits)
* **Q: When is the RBF kernel preferred over the Polynomial kernel?**  
  *A:* The RBF kernel maps features implicitly into an infinite-dimensional Hilbert space with only two hyperparameters ($C, \gamma$). The polynomial kernel can be numerically unstable when computing higher powers ($d \ge 5$).

### Experiment 12: K-Nearest Neighbors Image Classification (Digits)
* **Q: Why is KNN effective on 8x8 Digits but ineffective on high-resolution images?**  
  *A:* An 8x8 image has only 64 dimensions, where Euclidean distances retain relative spatial meaning. High-resolution images (e.g., $1024 \times 1024 \times 3$) suffer from the Curse of Dimensionality, where raw pixel distances fail to capture semantic content.

### Experiment 13: Multilayer Perceptron Topologies (Wine Quality)
* **Q: What is the trade-off of adding more hidden layers to an MLP?**  
  *A:* Adding hidden layers increases the network's capacity to learn hierarchical feature representations, but also increases training time and the risks of vanishing gradients and overfitting.

### Experiment 14: Neural Network Activation Dynamics (Sigmoid vs. Tanh vs. ReLU)
* **Q: Why does ReLU converge substantially faster than Sigmoid in deep neural networks?**  
  *A:* ReLU maintains a constant gradient of $1.0$ for all positive inputs ($z > 0$), preventing vanishing gradients and eliminating expensive exponential calculations during backpropagation.

### Experiment 15: Hyperparameter Tuning for MLP (GridSearchCV)
* **Q: How does `GridSearchCV` prevent test data leakage during hyperparameter tuning?**  
  *A:* It evaluates each parameter combination using internal cross-validation folds within the training set, keeping the test partition held out until final evaluation.

### Experiment 16: Hierarchical vs. K-Means Clustering (Iris)
* **Q: How do you choose the number of clusters from a Hierarchical Dendrogram?**  
  *A:* Find the longest vertical line in the dendrogram that is not intersected by any horizontal cluster-merging line, and draw a horizontal cutoff threshold across it.

### Experiment 17: K-Means Clustering & Elbow Method (Digits)
* **Q: Why does Inertia alone not identify the optimal number of clusters?**  
  *A:* Inertia decreases monotonically as $K$ increases, reaching $0$ when $K=N$ (every point is its own cluster). The Elbow Method identifies the point where additional clusters yield diminishing returns.

### Experiment 18: Resampling — Bootstrapping vs. 5-Fold Cross-Validation
* **Q: When is Bootstrapping preferred over Cross-Validation?**  
  *A:* Bootstrapping is preferred when calculating empirical confidence intervals and standard errors for small datasets where cross-validation folds would be too small to train reliably.

### Experiment 19: Ensemble Learning — Bagging vs. AdaBoost
* **Q: Why is AdaBoost more vulnerable to label noise than Bagging?**  
  *A:* AdaBoost increases the weights of misclassified instances exponentially in each round. If a misclassified instance is an outlier or has a noisy label, AdaBoost overfits to that noise, degrading generalization.

### Experiment 20: Bias-Variance Complexity Analysis
* **Q: What marks the optimal model complexity on a Bias-Variance plot?**  
  *A:* The minimum of the **Test (Validation) Error curve**. Beyond this point, training error continues downward while test error curves upward, signaling the onset of overfitting.

---

## Module 14: Python, NumPy, Pandas & Scikit-Learn Framework Essentials

### 14.1 What is the Scikit-Learn API contract (`fit`, `transform`, `predict`)?
* `fit(X, y)`: Estimates model parameters or dataset statistics from training data.
* `transform(X)`: Applies learned parameters to transform inputs (used by scalers, encoders, and imputers).
* `fit_transform(X, y)`: Computes statistics and applies the transformation in a single pass (training set only).
* `predict(X)`: Generates predicted labels or continuous values for new data using learned parameters.
* `score(X, y)`: Returns default evaluation metric ($R^2$ for regressors, Accuracy for classifiers).

### 14.2 What is the role of `make_pipeline()` in Scikit-Learn?
`make_pipeline()` chains data preprocessing transformers and estimators into a single composite object. During cross-validation, it fits transformers strictly on the training folds, preventing data leakage into the validation folds.

### 14.3 What is the difference between `loc` and `iloc` in Pandas?
* `loc`: Selects rows and columns by **label or boolean index** (e.g., `df.loc[0:5, 'MedInc']`).
* `iloc`: Selects rows and columns by **integer position index** (e.g., `df.iloc[0:5, 0:2]`).

### 14.4 What does `np.c_[]` do in NumPy?
`np.c_[]` concatenates 1D arrays or matrices along their second axis (column-wise). It is commonly used to prepend a column of ones to a feature matrix:
```python
X_design = np.c_[np.ones(len(X)), X]
```

### 14.5 What does `np.clip(z, -500, 500)` prevent in custom Logistic Regression?
Passing extreme negative inputs into the exponential function $e^{-z}$ triggers numeric floating-point overflow (`OverflowError: math range error`). Clipping values to $[-500, 500]$ stabilizes sigmoid calculations.

### 14.6 What is the difference between `coef_` and `intercept_` in linear models?
* `coef_`: The learned feature weights or slopes ($w_1, w_2, \dots, w_d$).
* `intercept_`: The learned bias term ($b$ or $w_0$), representing the expected output when all feature inputs equal zero.

### 14.7 What does `np.random.seed()` or `random_state` guarantee?
It initializes the pseudo-random number generator to a fixed state, ensuring that data splits, parameter initializations, and stochastic operations produce identical results across runs for deterministic reproducibility.
