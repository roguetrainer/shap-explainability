# Shapley Values: From Game Theory to Explainable AI

## 1. The History: A Game Theoretic Foundation

The concept of the Shapley value was introduced in **1951** by **Lloyd Shapley**. He sought to solve a specific problem in Cooperative Game Theory:

> *In a coalition of players working together to achieve a combined payout, how do we fairly distribute that payout based on each player's individual contribution?*

Shapley proved that there is **one unique way** to distribute the profit that satisfies specific axioms of fairness (Efficiency, Symmetry, Dummy, Additivity).

## 2. The Theory: Marginal Contributions

The core intuition is **marginal contribution**. To determine a player's worth, you ask: "How much did the group's output change when this player joined?"

Because players (features) interact, you theoretically must calculate the average marginal contribution of a player across **all possible permutations** (coalitions) of players.

### The Formula
The Shapley value $\phi_i$ for player $i$ is:

$$\phi_i(v) = \frac{1}{|N|!} \sum_{S \subseteq N \setminus \{i\}} |S|! (|N| - |S| - 1)! [v(S \cup \{i\}) - v(S)]$$

## 3. Applying Shapley Values to Machine Learning

In 2017, Lundberg and Lee mapped these concepts to AI:

| Game Theory Concept | Machine Learning Equivalent |
| :--- | :--- |
| **The Game** | The prediction task. |
| **The Players** | The input features. |
| **The Payout** | The prediction minus the average prediction. |

## 4. The SHAP Framework

Calculating exact Shapley values is NP-hard ($2^N$ combinations). **SHAP (SHapley Additive exPlanations)** makes this practical using approximations:

* **TreeSHAP:** Optimized for tree-based models (XGBoost, Random Forest).
* **DeepSHAP:** Adapted for Deep Learning.
* **KernelSHAP:** Model-agnostic method.
