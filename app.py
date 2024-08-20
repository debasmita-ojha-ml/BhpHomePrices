from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load the model and columns
model = joblib.load('model.pkl')
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']
    locations = [x for x in data_columns if x not in ['total_sqft', 'bath', 'bhk']]

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    loc_index = np.where(np.array(data_columns) == location)[0][0]

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]
    return render_template('index.html', prediction_text=f"Predicted Price: ₹{predicted_price:.2f} Lakhs", locations=locations)

if __name__ == "__main__":
    app.run(debug=True)
