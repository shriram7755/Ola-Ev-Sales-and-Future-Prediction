from flask import Flask, render_template, request
import dateutil
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import base64
import pickle
import io


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     years = int(request.form['years'])

#     # Load the data
#     df = pd.read_csv('data/ev_sales.csv')

#     # Load the trained SARIMA model
#     model = pickle.load(open('sarima_model.pkl', 'rb'))

#     # Forecast sales
#     forecast = model.forecast(steps=years * 12)

#     # Convert the forecast to a list
#     forecast_list = forecast.tolist()

#     # Convert the plot to base64
#     fig = plt.gcf()
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     plt.close()
#     plot_data = base64.b64encode(buf.getbuffer()).decode('ascii')
#     # Convert the plot buffer to base64
#     #plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')

#     # Render the result page
#     return render_template('result.html', plot_data=plot_data, forecast=forecast_list)


# @app.route('/predict', methods=['POST'])
# def predict():
#     years = int(request.form['years'])

#     # Load the data
#     df = pd.read_csv('data/ev_sales.csv')

#     # Load the trained SARIMA model
#     model = pickle.load(open('sarima_model.pkl', 'rb'))

#     # Forecast sales
#     forecast = model.forecast(steps=years * 12)

#     # Convert the forecast to a list
#     forecast_list = forecast.tolist()

#     # Plot the forecast
#     plt.figure(figsize=(10, 6))
#     plt.plot(forecast, label='Forecast')
#     plt.xlabel('Time')
#     plt.ylabel('Sales Quantity')
#     plt.title('SARIMA Forecast')
#     plt.legend()

#     # Convert plot to base64
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plot_data = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close()

#     # Render the result page
#     return render_template('result.html', plot_data=plot_data, forecast=forecast_list)


from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict():
    years = int(request.form['years'])

    # Load the trained SARIMA model
    model = pickle.load(open('sarima_model.pkl', 'rb'))

    # Forecast sales
    forecast = model.forecast(steps=years * 12)

    # Generate future dates for the forecast
    start_date = datetime.now().date()  # Assuming you want to start forecasting from the current date
    freq = 'M'  # Assuming you want monthly forecasts
    forecast_dates = pd.date_range(start=start_date, periods=years * 12, freq=freq)

    # Combine dates and forecast values into a DataFrame
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Sales': forecast})

    # Convert DataFrame to HTML table
    forecast_table = forecast_df.to_html(index=False)

    # Plot the forecast
    plt.figure(figsize=(8,5))
    plt.plot(forecast_dates, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Sales Quantity')
    plt.title('SARIMA Forecast')
    plt.legend()

    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    #buf.close()  # Close the buffer to release resources
    plt.close()

    # Render the result page
    return render_template('result.html', plot_data=plot_data, forecast_table=forecast_table)







if __name__ == '__main__':
    app.run(debug=True)



