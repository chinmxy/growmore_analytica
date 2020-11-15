from flask import Flask, request, render_template,json

app = Flask(__name__)

xdata = [20, 10, 25, 12, 20, 12, 25]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def accept_input():
    company_name = request.form['company-name']
    investment_amt = request.form['investment-amt']
    print(company_name, investment_amt)
    return render_template("result.html" , xdata = json.dumps(xdata))
    
@app.route('/news')
def news():
    return render_template("news.html")

@app.route('/about')
def about():
    return render_template("about.html")


app.run(debug=True)