import numpy as np
from flask import Flask, request, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('C:/Users/dell/Desktop/Python files/models/model.pk1', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    feature_names = ['Critical Micelle Concentration (CMC) (M)','Hydrophilic - Lipophilic Balance (HLB)','Solubility Ratio (SR)','Molecular Packing Parameter (MPP)','Density (g/mL)','Molecular Weight (g/mol)']
    model.feature_names = feature_names
    int_feature_names = [float(x) for x in request.form.values()]
    feature_names = [np.array(int_feature_names)]
    prediction = model.predict(feature_names)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text= 'The crude oil/brine IFT is {} mN/m' .format(output))

if __name__ == '__main__':
    app.run()
    

