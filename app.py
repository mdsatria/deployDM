from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('model.h5')

# home page
@app.route('/')
def index():
    return render_template('home.html')

# prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # if user submit prediction
    if request.method == 'POST':   
        # get data from form
        sepalLength = request.form.get('sepalLength') 
        sepalWidth = request.form.get('sepalWidth') 
        petalLength = request.form.get('petalLength') 
        petalWidth = request.form.get('petalWidth') 
        # scale data
        # mean and std
        atribut_mean = np.array([5.84333333, 3.05733333, 3.758 , 1.19933333])
        std_mean = np.array([0.82530129, 0.43441097, 1.75940407, 0.75969263])        
        sepalLength = (np.float(sepalLength) - atribut_mean[0])/ std_mean[0]
        sepalWidth = (np.float(sepalWidth) - atribut_mean[1])/ std_mean[1]
        petalLength = (np.float(petalLength) - atribut_mean[2])/ std_mean[2]
        petalWidth = (np.float(petalWidth) - atribut_mean[3])/ std_mean[3]
        # combine feature into 2d array
        feature = np.array([[sepalLength, sepalWidth, petalLength, petalWidth]]).astype(np.float16)
        # predict with loaded model
        prediction = model.predict(feature)
        # find the index of max value in vector prediction
        prediction = np.argmax(prediction,axis=1)
        target_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        result = target_names[int(prediction)]
        # return to predict.html page with prediction_text data
        return render_template('predict.html', prediction_text='{}'.format(result), img='{}'.format(result+'.jpg'))
    # default prediction page
    else:
        return render_template('predict.html')    

if __name__ == '__main__':
    app.run(debug=True)