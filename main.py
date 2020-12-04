from flask import Flask, request, render_template,json

from src import predict
from src.sentiment_analysis import sentiment as st

app = Flask(__name__)

# xdata = [20, 10, 25, 12, 20, 12, 25]


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def accept_input():
    company_name = request.form['company-name'].upper()
    investment_amt = request.form['investment-amt']
    # print(company_name, investment_amt)
    val1 = [20, 10, 25, 12, 20, 12, 25]
    val2 = [15, 12, 27, 10, 22, 15, 30]
    model_response = {}
    model_response["company_name"] = company_name
    model_response["investment_amt"] = investment_amt
    y_test, y_testLSTM, y_hatLSTM, y_hat_lin, y_hat_poly, y_hat_rbf, y_hat_rfr, y_hat_gbr, trade_lstm, trade_lin, trade_poly, trade_rbf, trade_rfr, trade_gbr = predict.predict_prices(company_name)
    model_response["real_valuesLSTM"] = y_testLSTM
    model_response["predicted_valuesLSTM"] = y_hatLSTM
    model_response["real_valuesML"] = y_test
    model_response["predicted_valuesSVR_Lin"] = y_hat_lin
    model_response["predicted_valuesSVR_Poly"] = y_hat_poly
    model_response["predicted_valuesSVR_Rbf"] = y_hat_rbf
    model_response["predicted_valuesRFR"] = y_hat_rfr
    model_response["predicted_valuesGBR"] = y_hat_gbr
    model_response["label"] = list(range(len(y_hatLSTM)))
    model_response["trade_lstm"] = trade_lstm
    model_response["trade_lin"] = trade_lin
    model_response["trade_poly"] = trade_poly
    model_response["trade_rbf"] = trade_rbf
    model_response["trade_rfr"] = trade_rfr
    model_response["trade_gbr"] = trade_gbr

    return render_template("result.html" , model_response = model_response)
    
@app.route('/news')
def news():
    return render_template("news.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/sentiment')
def sentiment():
    return render_template("sentiment.html")

@app.route('/sentiment', methods=["POST"])
def sentiment_post():
    company_name = request.form['company-name']
    article_length = int(request.form['article-length'])
    # print(company_name, article_length)
    st.calculate_mean_sentiment([company_name], article_length)
    return "Hellew"

app.run(debug=True)