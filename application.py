from flask import Flask, render_template, request, jsonify
import pandas as pd  # Ensure 'pd' alias is used consistently
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import ridge regression and standard scale pickle
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model or scaler pickle files not found. Make sure they are in the 'models' directory.")
    # Exit or handle the error gracefully if files are missing
    exit()

FEATURE_NAMES = [
    'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'
]
FORM_INPUT_ORDER = [
    'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC',  'ISI', 'Classes', 'Region'
]
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature')) if request.form.get(
            'Temperature') else None
        RH = float(request.form.get('RH')
                   ) if request.form.get('RH') else None
        Ws = float(request.form.get('Ws')
                   ) if request.form.get('Ws') else None
        Rain = float(request.form.get('Rain')
                     ) if request.form.get('Rain') else None
        FFMC = float(request.form.get('FFMC')
                     ) if request.form.get('FFMC') else None
        DMC = float(request.form.get('DMC')
                    ) if request.form.get('DMC') else None
        ISI = float(request.form.get('ISI')
                    ) if request.form.get('ISI') else None
        Classes = float(request.form.get('Classes')) if request.form.get(
            'Classes') else None
        Region = float(request.form.get('Region')
                       ) if request.form.get('Region') else None

        # Validate that all required fields were provided
        input_values = [Temperature, RH, Ws,
                        Rain, FFMC, DMC, ISI, Classes, Region]
        if any(val is None for val in input_values):
            missing_fields = [FORM_INPUT_ORDER[i]
                              for i, val in enumerate(input_values) if val is None]
            return render_template('home.html', results=f"Error: Missing or empty fields: {', '.join(missing_fields)}. Please provide all required numerical inputs."), 400
        new_data_df = pd.DataFrame(
            [input_values], columns=FORM_INPUT_ORDER)

        # Apply the scaler to the DataFrame
        new_data_scaled = standard_scaler.transform(new_data_df)

        # Make prediction
        result = ridge_model.predict(new_data_scaled)

        # Format for display
        return render_template('home.html', results=f"Predicted Output: {result[0]}")

    else:
        # For GET requests to /predictdata, simply render the home page (or a blank form)
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
