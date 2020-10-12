from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pickle

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
        sepalLength = (np.float(sepalLength) - 5.843)/ 0.825
        sepalWidth = (np.float(sepalWidth) - 3.057)/ 0.434
        petalLength = (np.float(petalLength) - 3.758)/ 1.759
        petalWidth = (np.float(petalWidth) - 1.199)/ 0.759
        # combine feature into 1d array
        feature = np.array([sepalLength, sepalWidth, petalLength, petalWidth]).astype(np.float16)
        # reshape feature to shape (1,4)
        feature = np.expand_dims(feature, axis=0)
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
    
# about page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)