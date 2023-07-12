from flask import Flask, request,render_template,jsonify
import numpy as np
import pandas as pd
import pickle

#loading_model
model = pickle.load(open('mdl_pkl','rb'))

#create_flask _app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    content = request.json

    sex = content['sex']
    length = content['length']
    diameter = content['diameter']
    height = content['height']
    wholeWeight = content['wholeWeight']
    Shuckedweight = content['Shuckedweight']
    Visceraweight = content['Visceraweight']
    Shellweight = content['Shellweight']
    #sex = int(request.form['sex'])
    #length = float(request.form['length'])
    #diameter = float(request.form['diameter'])
    #height = float(request.form['height'])
    #wholeWeight = float(request.form['wholeWeight'])
    #Shuckedweight = float(request.form['Shuckedweight'])
    #Visceraweight = float(request.form['Visceraweight'])
    #Shellweight = float(request.form['Shellweight'])

    features = np.array([[sex, length, diameter, height, wholeWeight, Shuckedweight, Visceraweight, Shellweight]])

    age = model.predict(features).reshape(1,-1)[0]
    # return {"age: " + str(age)}
    return jsonify({"age": age.tolist()})




#python main 
if __name__ == "__main__":
    app.run(debug=True)