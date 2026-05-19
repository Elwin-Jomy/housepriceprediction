import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("matches.csv")

# Keep required columns
df = df[[
    "HTGS",
    "ATGS",
    "HTGD",
    "ATGD",
    "DiffPts",
    "FTR"
]]

# Remove missing values
df = df.dropna()

# Clean FTR column
df["FTR"] = df["FTR"].astype(str).str.strip()

# Features
X = df[[
    "HTGS",
    "ATGS",
    "HTGD",
    "ATGD",
    "DiffPts"
]]

# Binary target
# H = Home Win
# NH = Not Home Win

y = df["FTR"].map({
    "H": 1,
    "NH": 0
})

# Remove invalid rows
valid_rows = y.notna()

X = X[valid_rows]
y = y[valid_rows]

# Train-test split
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

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nSample Predictions:")
print(predictions[:20])

print("\nClass Distribution:")
print(df["FTR"].value_counts())

# Save model and scaler
joblib.dump(model, "football_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully!")