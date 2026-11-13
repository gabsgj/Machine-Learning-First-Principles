# Module 10: Multilayer Perceptrons & Backpropagation

## Mathematical Foundations & Derivations

Artificial Neural Networks (ANNs) are universal function approximators capable of learning non-linear mappings $\hat{y} = f(x; \Theta)$ via compositions of affine transformations and non-linear activation functions.

---

### 1. Why a Single Perceptron Cannot Solve XOR
A single perceptron computes $f(x) = \text{step}(w_1 x_1 + w_2 x_2 + b)$. For the XOR truth table:
* $(0, 0) \to 0 \implies b < 0$
* $(0, 1) \to 1 \implies w_2 + b \ge 0$
* $(1, 0) \to 1 \implies w_1 + b \ge 0$
* $(1, 1) \to 0 \implies w_1 + w_2 + b < 0$

Adding inequalities 2 and 3: $(w_1 + b) + (w_2 + b) \ge 0 \implies w_1 + w_2 + 2b \ge 0$.
Since $b < 0$, $w_1 + w_2 + b > w_1 + w_2 + 2b \ge 0$, which contradicts inequality 4 ($w_1 + w_2 + b < 0$).
Thus, XOR is not linearly separable and requires at least one hidden layer.

---

### 2. Forward Propagation Equations
For layer $l \in \{1, \dots, L\}$:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$
where $a^{[0]} = x$ is the network input.

---

### 3. Backpropagation Derivation (The Chain Rule)
Let the loss function be Categorical Cross-Entropy:
$$\mathcal{L}(y, \hat{y}) = - \sum_{k=1}^K y_k \ln(\hat{y}_k)$$

1. **Output Layer Error $\delta^{[L]}$ (Softmax with Cross-Entropy):**
   $$\delta^{[L]} \equiv \frac{\partial \mathcal{L}}{\partial z^{[L]}} = a^{[L]} - y$$
2. **Hidden Layer Error $\delta^{[l]}$:**
   $$\delta^{[l]} \equiv \frac{\partial \mathcal{L}}{\partial z^{[l]}} = \left( (W^{[l+1]})^T \delta^{[l+1]} \right) \odot {g^{[l]}}'(z^{[l]})$$
3. **Weight & Bias Gradients:**
   $$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} \delta^{[l]} (a^{[l-1]})^T, \quad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^m \delta^{[l](i)}$$

---

### 4. Activation Functions & Gradient Saturation
| Activation | Equation $g(z)$ | Derivative $g'(z)$ | Range | Vanishing Gradient? |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $g(z)(1 - g(z))$ | $(0, 1)$ | **Severe** ($\max g' = 0.25$) |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - g(z)^2$ | $(-1, 1)$ | Moderate ($\max g' = 1.0$) |
| **ReLU** | $\max(0, z)$ | $\mathbb{I}(z > 0)$ | $[0, \infty)$ | **None** for $z > 0$ (Dying ReLU risk) |

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_mlp.py` | Full vectorised forward and backpropagation engine supporting arbitrary hidden layer configurations and activations. |
| `library_mlp.py` | Scikit-Learn `MLPClassifier` implementations covering Experiments 13 (Architectures), 14 (Activations), and 15 (Hyperparameter Tuning). |
| `compare_mlp_activations.py` | Side-by-side numerical verification and loss convergence curves for Sigmoid vs ReLU vs Tanh. |

## Quick Execution

```bash
python3 scratch_mlp.py
python3 library_mlp.py
python3 compare_mlp_activations.py
```
