import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

# Load your training dataset
train_file_path = 'add your train dataset here'  # Update this path
train_df = pd.read_csv(train_file_path)

# One-Hot Encoding for categorical features
categorical_features = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
    'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

# One-hot encode categorical features for the training dataset
train_df = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)

# Specify the relevant features and target variable
numeric_features = [
    'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
    'MoSold', 'YrSold'
]

features = numeric_features + list(train_df.columns.difference(numeric_features + ['SalePrice']))

target = 'SalePrice'

# Features and target variable for training
X_train = train_df[features]
y_train = train_df[target]

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Create a linear regression model
model = LinearRegression()

# Evaluate model with cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-Validated MAE: {-np.mean(cv_scores):,.2f}")

# Fit the model using scaled data
model.fit(X_train_scaled, y_train)

# Load your test dataset
test_file_path = 'add your test dataset here'  # Update this path
test_df = pd.read_csv(test_file_path)

# One-hot encode categorical features for the test dataset
test_df = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

# Ensure both train and test datasets have the same columns
train_columns = X_train.columns
test_df = test_df.reindex(columns=train_columns, fill_value=0)

# Check if 'Id' column exists in test_df before proceeding
if 'Id' not in test_df.columns:
    raise ValueError("The 'Id' column is missing from the test dataset.")

# Features for testing
X_test = test_df[features]

# Impute missing values in the test set using the same imputer
X_test_imputed = imputer.transform(X_test)

# Scale test features
X_test_scaled = scaler.transform(X_test_imputed)

# Make predictions on the test set
y_test_pred = model.predict(X_test_scaled)

# Check for negative predictions in the test set
negative_test_preds = (y_test_pred < 0).sum()
print(f"Number of negative predictions in test set: {negative_test_preds}")

# Handle negative predictions by setting them to zero
y_test_pred = np.maximum(y_test_pred, 0)

# Create a DataFrame with the IDs and the predicted prices
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_test_pred})

# Save the predictions to a CSV file
submission_file_path = '/content/submission.csv'  # Update this path
submission_df.to_csv(submission_file_path, index=False)

print(f'Predictions have been saved to {submission_file_path}')
