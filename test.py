import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load model and preprocessors
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

# ---------- PART 1: Predict on Custom Samples ----------
print("🔹 Predictions on Custom Samples:")
samples = np.array([
    [8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23],
    [3.1200, 15, 5.5, 1.1, 1500, 3.0, 34.25, -118.45],
    [7.9231, 52, 6.0, 1.0, 200, 2.0, 38.15, -121.40]
])
scaled_samples = scaler.transform(samples)
poly_samples = poly.transform(scaled_samples)
predictions = model.predict(poly_samples)

for i, pred in enumerate(predictions):
    print(f"Sample {i+1} predicted price: ${pred * 100000:.2f}")

# ---------- PART 2: Predict on Test Set and Compare ----------
print("\n🔹 Comparing Actual vs Predicted on Test Set:")

# Load dataset and split
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess test data
X_test_scaled = scaler.transform(X_test)
X_test_poly = poly.transform(X_test_scaled)

# Predict and compare
y_pred = model.predict(X_test_poly)
df_compare = pd.DataFrame({
    "Actual Price ($100k)": y_test,
    "Predicted Price ($100k)": y_pred
}).reset_index(drop=True).round(2)

# Save CSV
df_compare.to_csv("actual_vs_predicted.csv", index=False)
print("✅ Saved 'actual_vs_predicted.csv'")

# Display top 10 comparisons
print("\nSample Comparison:")
print(df_compare.head(10))

# ---------- PART 3: Plot ----------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price ($100k)")
plt.ylabel("Predicted Price ($100k)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_plot.png")
plt.show()
print("📊 Saved plot as 'prediction_plot.png'")
