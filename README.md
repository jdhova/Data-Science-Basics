# Supervised Learning

This folder contains hands-on Jupyter notebooks covering the fundamentals of supervised machine learning, applied to real-world financial crime and fraud-scoring datasets. Topics progress from simple linear regression through multiple linear regression and binary classification.

---

## Folder Structure

```
Superviced/
Γö£ΓöÇΓöÇ Classification/
Γöé   Γö£ΓöÇΓöÇ Classification.ipynb            # Logistic regression fundamentals
Γöé   Γö£ΓöÇΓöÇ Classification2.ipynb           # Bias-variance tradeoff & regularization
Γöé   Γö£ΓöÇΓöÇ important_features_dataset.csv
Γöé   ΓööΓöÇΓöÇ data/
Γöé       Γö£ΓöÇΓöÇ aml_fraud_customer_profiling.csv   # AML customer risk dataset
Γöé       ΓööΓöÇΓöÇ fraud_scoring_dataset.csv
Γö£ΓöÇΓöÇ Linear regression Basics/
Γöé   Γö£ΓöÇΓöÇ Linear Regression.ipynb         # Simple linear regression fundamentals
Γöé   ΓööΓöÇΓöÇ linear_regression_practice.csv  # Synthetic house-price dataset
ΓööΓöÇΓöÇ Multiple Linear Regression/
    Γö£ΓöÇΓöÇ MLR.ipynb                        # Dataset creation & enrichment
    Γö£ΓöÇΓöÇ MLR2.ipynb                       # Feature engineering & MLR model
    Γö£ΓöÇΓöÇ MLR3.ipynb                       # Encoding & classification setup
    ΓööΓöÇΓöÇ data/
        ΓööΓöÇΓöÇ fraud_scoring_dataset.csv    # Synthetic fraud-scoring dataset
```

---

## 1. Linear Regression Basics

**Notebook:** `Linear regression Basics/Linear Regression.ipynb`

### Dataset
A synthetic house-price dataset (`linear_regression_practice.csv`) generated with `numpy`:

| Column | Description |
|---|---|
| `size_sqft` | House size in square feet |
| `bedrooms` | Number of bedrooms (1ΓÇô5) |
| `age_years` | House age in years (0ΓÇô30) |
| `price` | Target ΓÇö house price |

True relationship used for generation:
$$\text{price} = 50{,}000 + 120 \times \text{size} + 15{,}000 \times \text{bedrooms} - 1{,}000 \times \text{age} + \epsilon$$


---

## Folder Structure

---

#### The Linear Equation
$$y = m \cdot x + b$$

Where $m$ is the slope (price change per sqft) and $b$ is the intercept (base price).

#### Cost Function ΓÇö Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

The notebook demonstrates why the best-fit line minimises MSE by comparing multiple candidate lines with different slopes and intercepts.

#### Closed-Form Solution
Using `np.polyfit`, the optimal slope and intercept are found analytically (the global minimum of the cost surface).

#### Gradient Descent (From Scratch)
Manual implementation showing:

- **Initialisation:** $m = 0$, $b = 0$
- **Prediction:** $\hat{y} = m \cdot x + b$
- **Gradients:**
$$\frac{\partial \text{MSE}}{\partial m} = \frac{2}{n} \sum (\hat{y} - y) \cdot x \qquad \frac{\partial \text{MSE}}{\partial b} = \frac{2}{n} \sum (\hat{y} - y)$$
- **Parameter updates:**
$$m \leftarrow m - \alpha \cdot \frac{\partial \text{MSE}}{\partial m} \qquad b \leftarrow b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}$$

The notebook runs 2,000 iterations with a learning rate of $\alpha = 1 \times 10^{-7}$ and produces a convergence table at key checkpoints (iterations 0, 1, 2, 5, 10, 50, 100, 200, 500, 1000, 1500, 2000).

#### Evaluation Metrics

| Metric | Formula |
|---|---|
| MSE | $\frac{1}{n} \sum (\hat{y} - y)^2$ |
| RMSE | $\sqrt{\text{MSE}}$ |
| MAE | $\frac{1}{n} \sum |\hat{y} - y|$ |
| R┬▓ | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ |

#### Visualisations
- Scatter plot with best-fit line
- Cost function surface: MSE vs slope (bowl-shaped convex curve)
- Candidate lines comparison (multiple slopes vs data)
- Gradient descent loss convergence over iterations
- Comparison table: gradient descent vs closed-form results

---

## 2. Multiple Linear Regression

### Notebook 1 ΓÇö Dataset Creation & Enrichment

**Notebook:** `Multiple Linear Regression/MLR.ipynb`

This notebook builds the `fraud_scoring_dataset.csv` used throughout the MLR section.

#### Synthetic Dataset (1,000 samples)

| Column | Description |
|---|---|
| `transaction_amount` | Transaction amount (10ΓÇô10,000) |
| `customer_age` | Customer age (18ΓÇô80) |
| `account_age_days` | Account age in days (1ΓÇô3,650) |
| `num_transactions_30days` | Transaction count in last 30 days |
| `failed_login_attempts` | Failed login count (0ΓÇô10) |
| `international_transaction` | Binary flag (0 or 1) |
| `fraud_score` | Target ΓÇö continuous score (0ΓÇô100) |

#### Data Enrichment
- **Dummy names** assigned randomly (`first_name`, `last_name`)
- **Country assignment** based on `fraud_score` bands:
  - High score (ΓëÑ66): safe countries (e.g. Germany, Sweden, Switzerland)
  - Medium score (33ΓÇô66): medium-risk countries (e.g. India, Brazil, Turkey)
  - Low score (<33): high-risk countries (e.g. Afghanistan, Yemen, Somalia)
- **FATF-based sanctioned countries flag** (`is_sanctioned_country`): Iran, North Korea, Russia, Syria, Cuba
- **Country risk group column**: `safe_country_group`, `medium_risk_country_group`, `high_risk_country_group`, `sanctioned_country_group`

---

### Notebook 2 ΓÇö Feature Engineering & MLR Model

**Notebook:** `Multiple Linear Regression/MLR2.ipynb`

#### Feature Selection
Works with `transaction_amount`, `country`, `customer_age` ΓåÆ target: `fraud_score`.

#### Correlation Analysis
- **Pearson correlation** between numeric features and `fraud_score`
- **Eta-squared** ($\eta^2$) to measure the categorical variable `country`'s effect on `fraud_score`

#### Z-Score Normalisation
$$z = \frac{x - \mu}{\sigma}$$

Applied to `transaction_amount` and `customer_age` before modelling. Verifies that normalised columns have mean Γëê 0 and std Γëê 1.

#### One-Hot Encoding
`pd.get_dummies` applied to the `country` column (29 countries), producing a wide encoded DataFrame ready for regression.

#### Multiple Linear Regression ΓÇö Normal Equation
The model formula:
$$f(x) = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

Solved analytically using the Normal Equation (least-squares via `np.linalg.lstsq`):
$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

80/20 train-test split. Top 10 feature weights printed by magnitude.

#### Gradient Descent for MLR
Visualises cost function decrease across multiple learning rates, showing the effect of $\alpha$ on convergence speed.

#### Visualisations
- Actual vs Predicted fraud score scatter plot (with R┬▓ annotation)
- Residuals vs Predicted scatter plot (with RMSE annotation)
- MSE, RMSE, MAE, R┬▓ summary table
- Cost curve across gradient descent iterations for different learning rates

#### Performance Table
Per-row comparison of actual vs predicted fraud score including residual, absolute error, percentage error, and accuracy.

---

### Notebook 3 ΓÇö Encoding & Classification Setup

**Notebook:** `Multiple Linear Regression/MLR3.ipynb`

Picks up the same `fraud_scoring_dataset.csv` and demonstrates:

- Z-Score Normalisation on `transaction_amount` and `customer_age`
- One-Hot Encoding of `country` column (29 countries, `drop_first=False`)
- Exploration of `country_risk_group` value counts ΓÇö framing the dataset as a multi-class classification problem (safe, medium-risk, high-risk, sanctioned)

---

## 3. Classification

### Notebook 1 ΓÇö Logistic Regression Fundamentals

**Notebook:** `Classification/Classification.ipynb`

#### Dataset
`data/aml_fraud_customer_profiling.csv` ΓÇö Anti-Money Laundering (AML) customer risk profiling.

Key columns used:

| Column | Description |
|---|---|
| `Risk_Target` | Binary target: 1 = High Risk, 0 = Low Risk |
| `Risk_Label` | Categorical risk label |
| `Country_Risk_Score` | Numeric country risk score |
| `Country_Risk_Score_Squared` | Squared country risk score (non-linear feature) |
| `Country_Risk_Flag` | Binary country risk flag |
| `Volume_x_CountryRisk` | Transaction volume ├ù country risk interaction |
| `Amount_x_CountryRisk` | Transaction amount ├ù country risk interaction |

#### Feature Selection by Correlation
Top 6 features selected by absolute Pearson correlation with `Risk_Target`.

#### The Sigmoid Function
$$p = \frac{1}{1 + e^{-(b_0 + b_1 x)}}$$

The notebook shows the full pipeline:
1. Compute linear score (log-odds): $z = b_0 + b_1 x$
2. Pass through sigmoid to get probability $p \in (0, 1)$
3. Classify as High Risk if $p \geq 0.5$
4. Decision boundary at the $x$ value where $z = 0$ (i.e. $x = -b_0 / b_1$)

#### Log Loss (Cross-Entropy)
Per-sample loss:
$$\mathcal{L}_i = -\left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]$$

Aggregate cost:
$$J = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i$$

#### Visualisations
- Sigmoid curve with decision boundary and 0.5 threshold
- Log-odds (linear score) plot
- Theoretical log-loss curves for $y=1$ and $y=0$ vs predicted probability
- Per-sample loss scatter (colour-coded by class)
- Sorted sample losses and cumulative mean converging to cost

#### Gradient Descent for Logistic Regression (From Scratch)
Manual implementation over 300 iterations ($\alpha = 0.1$, feature standardised):

- Update rule: $w \leftarrow w - \alpha \cdot \frac{1}{n} \sum (p_i - y_i) \cdot x_i$
- Tracks: cost history, weight history, gradient norm
- Plots: cost convergence curve and final learned sigmoid curve on the original feature scale

---

### Notebook 2 ΓÇö Bias-Variance Tradeoff & Regularisation

**Notebook:** `Classification/Classification2.ipynb`

Same AML dataset, builds on Classification.ipynb.

#### Underfitting vs Overfitting vs Just Right

| Model | Features | Complexity | Behaviour |
|---|---|---|---|
| **Underfitting** | 1 feature: `Country_Risk_Score` | Too simple (C=0.01) | Poor on both train & test |
| **Just Right** | 9 balanced features | Moderate (C=1.0) | Good on both train & test, small gap |
| **Overfitting** | All numeric features + degree-3 polynomial | Too complex (C=1e6) | High train accuracy, drops on test |

The gap $= \text{Train Accuracy} - \text{Test Accuracy}$ is used as the primary overfitting indicator.

#### L2 Regularisation (Ridge)
Regularised logistic regression cost function:
$$J(\theta) = -\frac{1}{m} \sum \left[ y \log(h_\theta(x)) + (1-y) \log(1 - h_\theta(x)) \right] + \frac{\lambda}{2m} \sum_{j} \theta_j^2$$

In scikit-learn, $C \approx \frac{1}{\lambda}$:
- **Small C** ΓåÆ strong regularisation ΓåÆ simpler model ΓåÆ risk of underfitting
- **Large C** ΓåÆ weak regularisation ΓåÆ complex model ΓåÆ risk of overfitting

#### Regularisation Sweep
13 values of C on a log scale ($10^{-4}$ to $10^4$) tested with degree-2 polynomial features. Tracks:
- Train accuracy, test accuracy, gap
- L2 norm of coefficients (shows how regularisation shrinks weights)
- Best C selected by highest test accuracy with smallest gap

#### Visualisations
- Train vs test accuracy vs log(C) ΓÇö highlights optimal zone
- Train-test gap vs log(C)
- Coefficient L2 norm vs log(C)
- Cost function formula panel with interpretation notes

---

## Datasets Summary

| Dataset | Location | Rows | Key Columns |
|---|---|---|---|
| `aml_fraud_customer_profiling.csv` | `Classification/data/` | ΓÇö | `Risk_Target`, `Risk_Label`, `Country_Risk_Score`, country-amount interactions |
| `linear_regression_practice.csv` | `Linear regression Basics/` | 200 | `size_sqft`, `bedrooms`, `age_years`, `price` |
| `fraud_scoring_dataset.csv` | `Multiple Linear Regression/data/` | 1,000 | `transaction_amount`, `customer_age`, `fraud_score`, `country`, `country_risk_group` |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical computing, gradient descent, matrix operations |
| `pandas` | Data loading, manipulation, feature engineering |
| `matplotlib` | Plots and visualisations |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | `LogisticRegression`, `LinearRegression`, `train_test_split`, `StandardScaler`, `PolynomialFeatures`, metrics |
| `openpyxl` | Excel file I/O |

---

## Key Learning Outcomes

1. Derive and implement the MSE cost function and understand how the best-fit line minimises it.
2. Implement gradient descent from scratch for both linear and logistic regression.
3. Understand the sigmoid function and how log-odds connect linear models to probabilities.
4. Apply Z-score normalisation and one-hot encoding as preprocessing steps for regression.
5. Build a Multiple Linear Regression model using the Normal Equation and interpret feature weights.
6. Demonstrate overfitting and underfitting empirically and use L2 regularisation to balance the bias-variance tradeoff.
7. Apply all concepts in the context of financial crime ΓÇö AML risk scoring and fraud detection.
