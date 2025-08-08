import os
print("Current directory:", os.getcwd())
print("Files here:", os.listdir())


from flask import Flask, request, jsonify
from H2O_model_loader import predict_greatest_absorption

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, world!"

@app.route('/predict')
def predict():
    try: 
        wn = float(request.args.get('wavenumber'))
        log10, real_result = predict_greatest_absorption(wn)
        return jsonify({'Absorption Coefficient (log10)': log10,
                        'Standard Absorption Coefficient' : real_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

import os

port = int(os.environ.get("PORT", 10000))
        
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)
