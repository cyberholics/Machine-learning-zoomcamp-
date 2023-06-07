import pickle
model_file="model_C={}.bin"
with open(model_file,"rb") as f_in:
    dv,model=pickle.load(f_in)
    
    
# the saved model will be loaded
customer={'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85}
X=dv.transform([customer])
y_pred=model.predict_proba(X)[:,1]
print("input",customer)
print("churn prediction",y_pred)
## serving the churn model as a flask app 
import pickle
import numpy as np

### serving the churn model as a flask app 
from flask import Flask, request, jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)