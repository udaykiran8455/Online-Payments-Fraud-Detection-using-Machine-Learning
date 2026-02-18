import os
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# IBM Tip: Use relative paths for the model so it works in the container
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'payments.pkl')

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
except FileNotFoundError:
    print("Error: payments.pkl not found. Ensure it's in the same directory.")

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/predict")
def predict_page():
    return render_template('predict.html')

@app.route("/pred", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Processing inputs
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        
        result = "FRAUD DETECTED" if prediction[0] == 1 else "TRANSACTION SECURE"
        return render_template('submit.html', prediction_text=result)

if __name__ == "__main__":
    # IBM Cloud Requirement: Get PORT from environment or default to 8080
    port = int(os.environ.get("PORT", 8080))
    # host='0.0.0.0' is required for the cloud to access the container
    app.run(host='0.0.0.0', port=port)