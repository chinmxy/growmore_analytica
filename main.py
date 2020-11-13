from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
    
@app.route('/news')
def news():
    return render_template("news.html")

app.run(debug=True)