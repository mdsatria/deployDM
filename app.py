from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('home.html')

# about page
@app.route('/about')
def about():
    return render_template('about.html')

# prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # if user submit prediction
    if request.method == 'POST':    
        # open model
        with open("model.pkl", "rb") as file:
            clf = pickle.load(file)
        # gather all input data into float-type list
        int_features = [float(x) for x in request.form.values()]
        # convert list into 2d numpy array
        int_features = np.array([int_features])    
        # predict based on saved model and only return value
        prediction = clf.predict(int_features)[0]
        # label name reference
        target_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        # get label name based on index from prediction variable
        hasil = target_names[prediction]
        # return to predict.html page with prediction_text data
        return render_template('predict.html', prediction_text='{}'.format(hasil), img='{}'.format(hasil+'.jpg'))
    # default prediction page
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)