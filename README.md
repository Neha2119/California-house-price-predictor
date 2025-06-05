# California-house-price-predictor

Flask app to predict California house prices using Ridge regression model.

# California House Price Predictor

## Overview

This project is a Flask-based web application that predicts house prices in California based on user input features such as number of bedrooms, bathrooms, square footage, and location. The app uses a machine learning model trained on the California Housing dataset to provide real-time price predictions.

The goal of this project is to demonstrate skills in:

* Building a full-stack web application using Flask
* Implementing machine learning models for regression tasks
* Integrating user inputs with predictive models
* Providing a clean and user-friendly interface for prediction

---

## Features

* Responsive web form to input house features
* Real-time house price prediction using a Ridge Regression model with polynomial features and scaling
* Clean UI with background image and styling
* Includes saved trained model and transformers for quick predictions
* Screenshots of the app in action included in the repository

---

## Demo

Access the app locally by running it on your machine (see Setup & Run below).
Screenshots are available in the `screenshots/` folder demonstrating form submission and prediction output.

---

## Setup & Run Locally

### Prerequisites

* Python 3.8 or above installed
* Git installed
* Recommended: Create and activate a Python virtual environment

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Neha2119/California-house-price-predictor.git
   cd California-house-price-predictor
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate       # For Windows
   source venv/bin/activate     # For Linux/macOS
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:

   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🔧 How to Use

1. **Run the Flask Application**
   Open your terminal and run:

   ```bash
   python app.py
   ```

2. **Access the App in Browser**
   Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser.

3. **Enter Sample Input Values in the Form**
   Example:

   * `MedInc`: `8.3252`
   * `HouseAge`: `41`
   * `AveRooms`: `6`
   * `AveBedrms`: `1`
   * `Population`: `322`
   * `AveOccup`: `2`
   * `Latitude`: `37.88`
   * `Longitude`: `-122.23`

4. **Submit and View Prediction**
   Click the **"Predict Price"** button.
   ✅ The predicted price will be displayed.
   For example, the output might be:

   ```
   Predicted House Price: $400823.56
   ```

---

## Project Structure

```
California-house-price-predictor/
│
├── app.py                  # Flask app with routes and prediction logic
├── model.pkl               # Trained Ridge Regression model
├── poly.pkl                # Polynomial features transformer
├── scaler.pkl              # Scaler object
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── templates/              # HTML templates
│   └── index.html
│
├── static/                 # CSS, images, and other static files
│   ├── style.css
│   └── background.jpg
│
├── screenshots/            # Screenshots of app in action
│   ├── Sample 1/
│   └── Sample 2/
│
├── train_model.py          # Script used to train the model (optional)
└── test.py                 # Testing script (optional)
```

---

## Screenshots

Screenshots of sample form submissions and predicted outputs are located in the `screenshots/` folder.

---

## Notes

* The model uses polynomial features and scaling to improve prediction accuracy.
* The app handles basic input validation for numerical values.
* Feel free to expand or modify the model or UI for further enhancements.

---

## Contact

For any questions or feedback, please contact me at \nehakothavade21@gmail.com
.
