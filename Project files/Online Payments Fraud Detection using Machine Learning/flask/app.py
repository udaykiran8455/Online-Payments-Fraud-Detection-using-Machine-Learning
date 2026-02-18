from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('payments.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict")
def predict_page():
    return render_template('predict.html')

@app.route("/pred", methods=['POST'])
def predict():
    # Extract values from form
    # Order: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    prediction = model.predict(final_features)
    
    result = "FRAUD DETECTED" if prediction[0] == 1 else "TRANSACTION SECURE"
    return render_template('submit.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)