from flask import Flask,render_template,url_for
app = Flask(__name__)


posts =[
	{
		'author':'vk',
		'title':'blogs 1'
	},
	{
		'author':'rk',
		'title':'blogs 2'
	}
]


@app.route("/")
@app.route("/home")
def hello():
	return render_template('home.html',posts=posts)


@app.route("/about")
def about():
	return render_template('about.html',title='About')





if __name__ == '__main__':
	app.run(debug=True)
