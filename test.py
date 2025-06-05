import joblib
import numpy as np

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

samples = np.array([
    [8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23],
    [3.1200, 15, 5.5, 1.1, 1500, 3.0, 34.25, -118.45],
    [7.9231, 52, 6.0, 1.0, 200, 2.0, 38.15, -121.40]
])

scaled = scaler.transform(samples)
poly_features = poly.transform(scaled)
predictions = model.predict(poly_features)

for i, pred in enumerate(predictions):
    print(f"Sample {i+1} predicted price: ${pred * 100000:.2f}")
