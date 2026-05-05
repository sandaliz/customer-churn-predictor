# Data Preprocessing Documentation

## Before & After

| | Before | After |
|---|--------|-------|
| Rows | 7,043 | 7,032 |
| Columns | 21 | 31 |
| Target | "Yes"/"No" | 1/0 |
| Features | Unscaled | Scaled (mean=0, std=1) |

---

## Steps Performed

### 1. Fix TotalCharges
- **Problem**: Stored as string
- **Fix**: Converted to numeric, dropped 11 empty rows
- **Why**: Models need numbers, not text

### 2. Drop customerID
- **Problem**: Unique identifier for each customer
- **Fix**: Removed column
- **Why**: Prevents overfitting (model would memorize instead of learn)

### 3. Encode Target (Churn)
- **Problem**: "Yes"/"No" text
- **Fix**: Mapped to 1/0 (Yes=1, No=0)
- **Why**: Models need numeric targets

### 4. One-Hot Encode Categories
- **Problem**: 15 columns had text categories (Contract, PaymentMethod, etc.)
- **Fix**: Converted each category to binary column
- **Why**: Model can't understand "Month-to-month" > "One year" (wrong order)

### 5. Train/Test Split
- **Split**: 80% train (5,625), 20% test (1,407)
- **Stratify**: Yes (both sets have 26.6% churn rate)
- **Why**: Test set must be unseen to evaluate real performance

### 6. Scale Features
- **Method**: StandardScaler (mean=0, std=1)
- **Fit on**: TRAIN only (no data leakage)
- **Why**: Logistic Regression needs features on same scale

### 7. Save Artifacts
- **Files saved**: scaler, feature_names, train/test sets
- **Why**: Can reuse preprocessing for new predictions

---

## Files Saved in `/models/`

| File | Shape | Purpose |
|------|-------|---------|
| scaler.pkl | - | Scale new data |
| feature_names.pkl | 30 names | API input validation |
| X_train_scaled.npy | 5,625 x 30 | Train models |
| X_test_scaled.npy | 1,407 x 30 | Test models |
| y_train.npy | 5,625 | Train labels |
| y_test.npy | 1,407 | Test labels |

---

## Ready for Modeling

Models to train:
- Logistic Regression (baseline)
- Decision Tree (interpretable)
- Random Forest (best for production)