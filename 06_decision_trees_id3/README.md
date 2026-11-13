# Module 06: Decision Trees (ID3 & CART)

## Mathematical Foundations & Derivations

Decision Trees construct recursive partitioning of the feature space. The **ID3 (Iterative Dichotomiser 3)** algorithm applies a greedy top-down heuristic governed by Information Theory.

---

### 1. Shannon Entropy
Entropy measures the fundamental uncertainty or impurity of a set of samples $S$:
$$H(S) = - \sum_{c=1}^C p_c \log_2(p_c)$$
where $p_c = \frac{|S_c|}{|S|}$ is the proportion of instances belonging to class $c$.
* If all instances belong to the same class: $p=1 \implies H(S) = 0$ (Pure Node).
* For binary classification with uniform distribution ($p=0.5$): $H(S) = 1.0$ bit (Maximum Uncertainty).

---

### 2. Information Gain
Information Gain quantifies the expected reduction in entropy achieved by partitioning dataset $S$ along attribute $A$:
$$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
where $S_v = \{x \in S \mid A(x) = v\}$. At each recursive step, ID3 selects the attribute:
$$A^* = \arg\max_{A} \text{Gain}(S, A)$$

---

### 3. Gini Impurity (CART)
The CART algorithm uses Gini impurity instead of entropy:
$$\text{Gini}(S) = 1 - \sum_{c=1}^C p_c^2$$
Gini computation avoids transcendental logarithm evaluations, offering faster execution during large-scale tree construction.

---

### 4. ID3 vs. Logistic Regression Comparison
| Dimension | Logistic Regression | Decision Tree (ID3 / CART) |
| :--- | :--- | :--- |
| **Decision Boundary** | Smooth linear hyperplanes ($w^T x + b = 0$) | Orthogonal, axis-aligned rectangular boxes |
| **Feature Scaling** | Strictly required for gradient stability | Invariant to monotonic scaling/transformations |
| **Interpretability** | Global linear feature weights | Local hierarchical IF-THEN rules |
| **Interactions** | Requires explicit interaction terms ($x_1 x_2$) | Captures high-order feature interactions natively |

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_id3_decision_tree.py` | Recursive symbolic ID3 tree builder with Shannon entropy and Information Gain. |
| `library_decision_tree.py` | Scikit-Learn `DecisionTreeClassifier` with feature importances and tree diagram exporter. |
| `compare_decision_tree.py` | Benchmarks Scratch ID3 on Adult Census dataset and compares Logistic Regression vs Decision Tree head-to-head. |

## Quick Execution

```bash
python3 scratch_id3_decision_tree.py
python3 library_decision_tree.py
python3 compare_decision_tree.py
```
