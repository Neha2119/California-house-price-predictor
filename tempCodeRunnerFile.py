from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    # 1. Load Dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # 2. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Feature scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Polynomial features to capture non-linearity (degree=2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # 5. Train Ridge Regression with regularization
    model = Ridge(alpha=1.0)
    model.fit(X_train_poly, y_train)

    # 6. Predict on test data
    y_pred = model.predict(X_test_poly)

    # 7. Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Improved Model Evaluation:")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - R² Score: {r2:.4f}")

    # 8. Save the trained model, scaler and poly objects
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(poly, 'poly.pkl')

    print("✅ Model, scaler, and polynomial transformer saved.")
    



if __name__ == "__main__":
    main()
