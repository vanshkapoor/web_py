from flask import Flask,url_for,render_template
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict' ,methods=['POST'])
def predict():
	df= pd.read_csv("names_dataset.csv")
	# Features and Labels
	df_X = df.name
	df_Y = df.sex
    
    # Vectorization
	corpus = df_X
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) 
	
	# Loading our ML Model
	naivebayes_model = open("models/naivebayesgendermodel.pkl","rb")
	clf = joblib.load(naivebayes_model)

	# Receives the input query from form
	if request.method == 'POST':
		name = request.form['name']
		data = [name]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction,name = name.upper())


if __name__ == '__main__':
	app.run(debug=1)


