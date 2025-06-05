from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and transformers
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    error = None

    if request.method == 'POST':
        try:
            # Input fields expected
            fields = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                      'Population', 'AveOccup', 'Latitude', 'Longitude']
            input_features = []

            for field in fields:
                value = request.form.get(field)
                if value is None or value.strip() == "":
                    raise ValueError(f"Missing input for '{field}'")

                try:
                    num = float(value)
                    # All values must be non-negative *except* Longitude
                    if field != 'Longitude' and num < 0:
                        raise ValueError(f"'{field}' cannot be negative.")
                    input_features.append(num)
                except ValueError:
                    raise ValueError(f"Invalid numeric input for '{field}'.")

            # Transform for prediction
            input_array = np.array(input_features).reshape(1, -1)
            scaled = scaler.transform(input_array)
            poly_features = poly.transform(scaled)

            prediction = model.predict(poly_features)[0]
            prediction_text = f"{prediction * 100000:.2f}"

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = "Unexpected error occurred. Please check your input values."

    return render_template('index.html', prediction_text=prediction_text, error=error)

if __name__ == '__main__':
    app.run(debug=True)
