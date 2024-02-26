from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pickled SARIMA model
with open('sarima_model.pkl', 'rb') as f:
    sarima_model = pickle.load(f)

# Function to generate SARIMA predictions and forecast
def generate_predictions_and_forecast(years):
    # Generate predictions
    predictions = sarima_model.predict(start=len(df), end=len(df)+years*12-1, typ='levels')
    
    # Generate forecast
    forecast = sarima_model.get_forecast(steps=years*12)
    
    return predictions, forecast

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    years = int(request.form['years'])
    predictions, forecast = generate_predictions_and_forecast(years)
    return render_template('results.html', predictions=predictions, forecast=forecast)

@app.route('/get_predictions_and_forecast')
def get_predictions_and_forecast():
    years = 5  # Default number of years for forecast
    predictions, forecast = generate_predictions_and_forecast(years)
    
    # Convert predictions and forecast to JSON format
    predictions_data = {'x': predictions.index.tolist(), 'y': predictions.tolist()}
    forecast_data = {'x': forecast.index.tolist(), 'y': forecast.predicted_mean.tolist()}
    
    return jsonify(predictions=predictions_data, forecast=forecast_data)

if __name__ == '__main__':
    app.run(debug=True)
