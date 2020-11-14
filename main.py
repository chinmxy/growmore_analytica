from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def accept_input():
    company_name = request.form['company-name']
    investment_amt = request.form['investment-amt']
    print(company_name, investment_amt)
    return render_template("result.html")
    
@app.route('/news')
def news():
    return render_template("news.html")

app.run(debug=True)