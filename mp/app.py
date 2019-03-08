from flask import Flask,url_for,request,render_template
import pandas as pd
#import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods =['POST'])
def predict():
	df = pd.read_csv("YoutubeSpamMergedData.csv")
	df_data = df[["CONTENT", "CLASS"]]

	df_x = df_data['CONTENT']
	df_y = df_data['CLASS']

	cv = CountVectorizer()
	corpus = df_x
	x = cv.fit_transform(corpus)

	from sklearn.model_selection import train_test_split
	x_train,x_test,y_train,y_test = train_test_split(x,df_y,test_size=0.33,random_state=42)

	from sklearn.naive_bayes import MultinomialNB
	clf=MultinomialNB()
	clf.fit(x_train,y_train)
	clf.score(x_test,y_test)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__=='__main__':
	app.run(debug=1)