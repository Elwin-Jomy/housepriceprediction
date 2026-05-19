import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv("train.csv")

# Select only 5 features
X = df[[
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "FullBath",
    "YearBuilt"
]]

y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GradientBoostingRegressor()

model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully!")