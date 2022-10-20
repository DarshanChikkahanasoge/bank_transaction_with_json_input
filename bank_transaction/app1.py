import pickle
from flask import Flask, request ,jsonify
from preprocessing import preprocess_output

app1=Flask(__name__)

@app1.route('/predict',methods=['POST'])
def predict():
    features=request.get_json()
    with open('bank_transaction/model.bin','rb') as f_in:
        model=pickle.load(f_in)
        f_in.close()
    predictions= preprocess_output(features)
    pred=model.predict(predictions).tolist()
    result = {
        'Predictions':pred

    }
    return jsonify(result)


if __name__=='__main__':
    app1.run(debug=True)