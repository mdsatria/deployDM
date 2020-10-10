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
        # ambil semua nilai dalam post ke dalam satu array
        int_features = [float(x) for x in request.form.values()]
        # convert list ke 2d numpy array
        int_features = np.array([int_features])    
        # lakukan prediksi dari model tersimpan dan kembalikan kode kelas
        prediction = clf.predict(int_features)[0]
        # referensi label kelas
        target_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        # cari nama label berdasarkan kode kelas hasil prediksi
        hasil = target_names[prediction]
        # return halaman predict.html dengan data hasil prediksi
        return render_template('predict.html', prediction_text='{}'.format(hasil), img='{}'.format(hasil+'.jpg'))
    # halaman default prediction jika user belum submit
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)