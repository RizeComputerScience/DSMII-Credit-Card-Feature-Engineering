# Credit Card Default Prediction - Feature Engineering Practice

This project demonstrates feature engineering techniques using the UCI Credit Card Default dataset. You'll learn how to create, test, and evaluate engineered features to improve machine learning model performance.

## 📁 Project Structure

```
Unit4-Practice/
├── data/
│   └── UCI_Credit_Card.csv          # Credit card default dataset
├── feature_engineering.ipynb         # Tutorial: Creating engineered features
├── model_creation.ipynb             # Testing feature impact on model performance
└── README.md                        # This file
```

## 🎯 Learning Objectives

1. **Understand Feature Engineering**: Learn different types of feature engineering techniques
   - Division (Ratios): Utilization rates, payment rates
   - Subtraction (Differences): Available credit, underpayment amounts
   - Boolean Features: Binary flags for specific conditions
   - Temporal Features: Payment history patterns over time

2. **Avoid Common Pitfalls**: Handle infinity values and edge cases in division operations

3. **Measure Impact**: Compare baseline vs. engineered models to determine if features add value

4. **Evaluate Significance**: Learn when improvements are meaningful vs. just noise

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn jupyter
```

### Dataset

The UCI Credit Card Default dataset contains:
- **30,000 customers** from Taiwan
- **24 features** including credit limit, age, payment history, bill amounts, and payment amounts
- **Target variable**: Whether the customer defaulted on their payment next month

## 📓 How to Run

### Part 1: Feature Engineering Tutorial (`feature_engineering.ipynb`)

This notebook teaches you how to create engineered features step-by-step:

1. **Open the notebook**:
   ```bash
   jupyter notebook feature_engineering.ipynb
   ```

2. **Run cells sequentially** (top to bottom):
   - Cell 1-4: Load data and libraries
   - Cell 5: Create baseline model with raw features only
   - Cell 6-8: Create ratio features (utilization_rate, payment_rate)
   - Cell 9-10: Create difference features (available_credit, underpayment)
   - Cell 11-12: Train model with engineered features
   - Cell 13-17: Create and test one-hot encoding
   - Cell 18-20: Create and test boolean features

3. **Expected baseline accuracy**: ~79.65%

4. **Key takeaway**: Not all engineered features improve the model - always compare to baseline!

### Part 2: Feature Impact Testing (`model_creation.ipynb`)

This notebook demonstrates how to properly test if engineered features improve your model:

1. **Open the notebook**:
   ```bash
   jupyter notebook model_creation.ipynb
   ```

2. **Run cells sequentially**:
   - Cell 1-3: Create baseline model with only raw features
   - Cell 4-5: Engineer temporal and interaction features
   - Cell 6-7: Train model with all features and compare
   - Cell 8: View side-by-side comparison and improvements
   - Cell 9: Examine feature importance rankings

3. **What to look for**:
   - **Accuracy improvement** > 1-2% → Likely significant
   - **Recall improvement** → Better at catching actual defaults
   - **Precision improvement** → Fewer false alarms
   - **Feature importance** → Which engineered features matter most

## 🔑 Key Concepts

### Handling Infinity Values

When creating ratio features, always handle division by zero or very small numbers:

```python
# ❌ BAD - Can create infinity
df['payment_rate'] = df['PAY_AMT1'] / df['BILL_AMT1']

# ✅ GOOD - Use np.where with threshold
df['payment_rate'] = np.where(
    df['BILL_AMT1'] > 100,
    df['PAY_AMT1'] / df['BILL_AMT1'],
    0
)

# Or replace infinity values after calculation
df['payment_rate'] = df['payment_rate'].replace([np.inf, -np.inf], 0)
```

### Comparing Models Fairly

Always use the **same random_state** when comparing models:

```python
# Baseline model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Engineered model - MUST use same random_state=42
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)
```

### When Are Improvements Significant?

- **< 0.5% improvement**: Likely just noise, not meaningful
- **0.5-1% improvement**: Borderline, needs cross-validation to confirm
- **> 1-2% improvement**: Likely significant and worth keeping
- **> 5% improvement**: Very strong signal, definitely keep the features!