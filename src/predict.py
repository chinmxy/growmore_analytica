import talib
#Import Libraries

import warnings
warnings.filterwarnings("ignore")
import sys
import os
import math
import pandas as pd
import numpy as np
import datetime
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import altair as alt
import seaborn as sns
import logging

from .trading_bot.trading_bot.agent import Agent
from .trading_bot.trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
from .trading_bot.trading_bot.methods import evaluate_model



# path_agent_data = "C:\\Users\\ethan\\Projects\\growmore_analytics_flask\\src\\trading_bot\\data"
# path_home_dir = "C:\\Users\\ethan\\Projects\\growmore_analytics_flask"

path_agent_data = "C:\\Personal\\Projects\\growmore_analytica\\src\\trading_bot\\data"
path_home_dir = "C:\\Personal\\Projects\\growmore_analytica"

def directional_asymmetry(y_hat, y_test):
  next_real = pd.Series(np.reshape(y_test, (y_test.shape[0]))).shift(-1)
  next_pred = pd.Series(np.reshape(y_hat, (y_hat.shape[0]))).shift(-1)
  curr_real = pd.Series(np.reshape(y_test, (y_test.shape[0])))[:y_test.shape[0] - 1]
  next_real.dropna(inplace=True)
  next_pred.dropna(inplace=True)
  direction_count = 0
  for i in range(next_real.shape[0]):
    if next_real[i] > curr_real[i] and next_pred[i] > curr_real[i]:
      direction_count += 1
    elif next_real[i] < curr_real[i] and next_pred[i] < curr_real[i]:
      direction_count += 1
    elif next_real[i] == curr_real[i] and next_pred[i] == curr_real[i]:
      direction_count += 1
  return 1 - (direction_count / next_real.shape[0])




def getMACD_MHIST(stock_close):
  #Compute MACD and MACD Histogram
  macd, macdsignal, macdhist = talib.MACD(stock_close, fastperiod=12, slowperiod=26, signalperiod=9)

  dict = {'MACD': macd, 'MSIG': macdsignal}
  macdata = []
  macdata = pd.DataFrame(data=dict)
  macdata.dropna(inplace=True)

  macdata['MACD_Signal1'] = macdata.apply(lambda x : 1 if x['MACD'] > x['MSIG'] else 0, axis = 1)

  n_days = len(macdata['MACD'])
  Signal = np.array(macdata['MACD_Signal1'])
  psy = []



  for d in range(0, n_days):
      
      if Signal[d] == 1:
          psycology = 1
          psy.append(psycology)
      
      elif Signal[d] == 0:
          psycology = 0
          psy.append(psycology)
          
  macdata['MACD_Signal'] = psy

  del macdata['MACD_Signal1']

  dict = {'MHIST': macdhist, 'PrevMHIST': macdhist.shift(1)}
  machdata = []
  machdata = pd.DataFrame(data=dict)
  machdata.dropna(inplace=True)

  machdata['MHIST_Signal1'] = machdata.apply(lambda x : 1 if x['MHIST'] > x['PrevMHIST'] else 0, axis = 1)

  n_days = len(machdata['MHIST'])
  Signal = np.array(machdata['MHIST_Signal1'])
  psy = []

  for d in range(0, n_days):
      
      if Signal[d] == 1:
          psycology = 1
          psy.append(psycology)
      
      elif Signal[d] == 0:
          psycology = 0
          psy.append(psycology)
          
  machdata['MHIST_Signal'] = psy

  del machdata['MHIST_Signal1']

  return macdata, machdata



def getRSI(stock_close):
  rsi = talib.RSI(stock_close, timeperiod=13)

  dict = {'Close': stock_close, 'RSI': rsi }

  rsidata = []
  rsidata = pd.DataFrame(data=dict)
  rsidata.dropna(inplace=True)

  rsidata['rsi1'] = rsidata.apply(lambda x : 1 if x['RSI'] < 30 else 0, axis=1)
  rsidata['rsi2'] = rsidata.apply(lambda x : -1 if x['RSI'] > 70 else 0, axis=1)
  rsidata['Sign1'] = rsidata.apply(lambda x : x['rsi1'] + x['rsi2'], axis=1)

  n_days = len(rsidata['RSI'])
  Signal = np.array(rsidata['Sign1'])
  psy = []

  for d in range(0, n_days):
      if Signal[d] == 1:
          psycology = 1
          psy.append(psycology)
      
      elif Signal[d] == -1:
          psycology = 0
          psy.append(psycology)
      
      elif Signal[d] == 0:
          psycology = 0
          psy.append(psycology)
          
  rsidata['Psycology'] = psy

  # del rsidata['Sign1']
  # del rsidata['rsi1']
  # del rsidata['rsi2']

  return rsidata


def getBB(stock_close):

  upper, middle, lower = talib.BBANDS(stock_close, timeperiod=26)

  dict = {'Close': stock_close, 'Middle': middle, 'Upper': upper, 'Lower': lower }

  bbdata = []
  bbdata = pd.DataFrame(data=dict)
  bbdata.dropna(inplace=True)

  #Generate the Long and Short Signals
  n_days = len(bbdata['Middle'])
  cash = 1
  stock = 0

  position = []

  spread = stock_close
  ma = middle
  upper_band = upper
  lower_band = lower

  for d in range(0, n_days):
      
      # Long if spread < lower band & if not bought yet
      if spread[d] < lower_band[d] and cash == 1:
          signal = 1
          cash = 0
          stock = 1
          position.append(signal)
          
          
      # Take Profit if spread > moving average & if already bought
      elif spread[d] > ma[d] and stock == 1:
          signal = 3
          cash = 1
          stock = 0
          position.append(signal)
          
      # Short if spread > upper band and no current position
      elif spread[d] > upper_band[d] and cash == 1:
          signal = -1
          cash = 0
          stock = -1
          position.append(signal)
          

      # Take Profit if spread < moving average & if already short
      elif spread[d] < ma[d] and stock == -1:
          signal = 3
          cash = 1
          stock = 0
          position.append(signal)
      
      else:
          signal = 0
          position.append(signal)
          
          
  bbdata['Position1'] = position
  bbdata['Position1'] = bbdata['Position1'].replace(to_replace=0, method= 'ffill')
  bbdata['Position1'] = bbdata['Position1'].replace(3,0)
  bbdata['Position'] = bbdata['Position1']

  del bbdata['Position1']

  t_days = len(bbdata['Middle'])
  Signal = np.array(bbdata['Position'])
  pos = []

  for d in range(0, t_days):
      if Signal[d] == 0:
          strategy = 0
          pos.append(strategy)
      
      elif Signal[d] == 1:
          strategy = 1
          pos.append(strategy)
      
      elif Signal[d] == -1:
          strategy = 0
          pos.append(strategy)

  bbdata['Strategy'] = pos

  return bbdata












def visualize(df, history, title="trading session"):
    # add history to dataframe
    position = [history[0][0]] + [x[0] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df['position'] = position
    df['action'] = actions
    
    # specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(min(df['actual']), min(df['position'])) - 50, max(max(df['actual']), max(df['position'])) + 50), clamp=True)
    

    # plot a line chart for stock positions
    actual = alt.Chart(df).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('position', axis=alt.Axis(format='.2f', title='Price (₹)'), scale=scale)
    ).interactive(
        bind_y=False
    )
    
    # plot the BUY and SELL actions as points
    points = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(format='.2f', title='Price (₹)'), scale=scale),
        color='action'
    ).interactive(bind_y=False)

    # merge the two charts
    chart = alt.layer(actual, points, title=title).properties(height=300, width=1000)
    
    return chart
    










def getChart(stock, model):
    os.chdir(path_agent_data)

    model_name = 'model_double-dqn_GOOG_50'
    test_stock = stock+'_'+model+'.csv'
    window_size = 10
    debug = True

    agent = Agent(window_size, pretrained=True, model_name=model_name)

    df = pd.read_csv(test_stock)
    df = df[['Date', 'Adj Close']]
    df = df.rename(columns={'Adj Close': 'actual', 'Date': 'date'})
    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df['date'] = dates

    switch_k_backend_device()

    test_data = get_stock_data(test_stock)
    initial_offset = test_data[1] - test_data[0]

    test_result, history = evaluate_model(agent, test_data, window_size, debug)
    show_eval_result(model_name, test_result, initial_offset)
    print(history)
    print(test_result)

    chart = visualize(df, history, title=model)
    return chart.to_json()







def predict_prices(stock_name):
    stock_name = stock_name
    end = datetime.datetime.today()
    start = datetime.date(end.year - 2, 1, 1)
    df = web.DataReader(stock_name+".NS", 'yahoo', start, end)

    # Create a new file for the data
    df.to_csv(stock_name+'_prices.csv')


    os.chdir(path_home_dir)

    # Read the csv file
    df = pd.read_csv(stock_name+'_prices.csv', date_parser=True)
    df_copy = df
    df = df[['Open', 'High', 'Low', 'Close']]
    df.dropna(inplace=True)

    stock_open = df['Open']
    stock_high = df['High']
    stock_low = df['Low']
    stock_close = df['Close']


    rsidata = getRSI(stock_close)
    macdata, machdata = getMACD_MHIST(stock_close)
    bbdata = getBB(stock_close)

    # Computing Next DataPoint Move
    stock_move = stock_close.shift(-1)

    dict = {'Close': stock_close, 'Move': stock_move}

    sdmdata = []
    sdmdata = pd.DataFrame(data=dict)
    sdmdata.dropna(inplace=True)

    sdmdata['sign'] = sdmdata.apply(lambda x : 1 if np.log(x['Move']/x['Close']) > 0 else -1, axis=1)


    n_days = len(sdmdata['Move'])
    Signal = sdmdata['sign']
    psy = []

    for d in range(0, n_days):
        if Signal[d] == 1:
            psycology = 1
            psy.append(psycology)
        
        elif Signal[d] == -1:
            psycology = 0
            psy.append(psycology)
        
        
    sdmdata['Next Point Move'] = psy

    del sdmdata['sign']

    Close = pd.DataFrame({'Close': stock_close})
    NM = pd.DataFrame({'NM' : sdmdata['Move']})
    RSI = pd.DataFrame({'RSI': rsidata['RSI']})
    MACD = pd.DataFrame({'MACD': macdata['MACD']})
    MHIST = pd.DataFrame({'MHIST': machdata['MHIST']})
    Middle = pd.DataFrame({'Middle' : bbdata['Middle']})
    Upper = pd.DataFrame({'Upper' : bbdata['Upper']})
    Lower = pd.DataFrame({'Lower' : bbdata['Lower']})
    # Merging into Single DataFrame

    merge1 = pd.merge(Close, NM, left_index=True, right_index=True, how='outer')
    merge2 = pd.merge(merge1, RSI, left_index=True, right_index=True, how='outer')
    merge3 = pd.merge(merge2, MACD, left_index=True, right_index=True, how='outer')
    merge4 = pd.merge(merge3, MHIST, left_index=True, right_index=True, how='outer')
    merge5 = pd.merge(merge4, Middle, left_index=True, right_index=True, how='outer')
    merge6 = pd.merge(merge5, Upper, left_index=True, right_index=True, how='outer')
    df_final = pd.merge(merge6, Lower, left_index=True, right_index=True, how='outer')
    df_final.dropna(inplace=True)


    training_size = math.ceil(0.90 * df_final.shape[0])

    test_size = df_final.shape[0] - training_size

    train_data = df_final[:training_size]

    X_train = train_data[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]
    y_train = train_data[['NM']]

    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()

    X_train = scalerX.fit_transform(X_train)
    y_train = scalerY.fit_transform(y_train)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_trainLSTM, y_trainLSTM = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)), np.array(y_train)


    model = Sequential()

    model.add(LSTM(units=120, activation='relu', return_sequences=True, input_shape=(X_trainLSTM.shape[1], 1)))
    model.add(Dropout(rate=0.1))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(rate=0.1))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(rate=0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_trainLSTM, y_trainLSTM, batch_size=128, epochs=30)

    regr_rbf = SVR(kernel='rbf', gamma=0.1)
    regr_poly = SVR(kernel='poly', degree=2)
    regr_lin = SVR(kernel='linear')
    regr_rfr = RandomForestRegressor(n_estimators=150, criterion='mse', oob_score=True)
    regr_gbr = GradientBoostingRegressor(loss='ls', n_estimators=150, criterion='friedman_mse', )

    regr_rbf.fit(X_train, y_train)
    regr_poly.fit(X_train, y_train)
    regr_lin.fit(X_train, y_train)
    regr_rfr.fit(X_train, y_train)
    regr_gbr.fit(X_train, y_train)

    test_data = df_final[training_size - 1:]

    X_test = test_data[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]
    y_test = test_data[['NM']]

    X_test = scalerX.fit_transform(X_test)
    y_test = scalerY.fit_transform(y_test)

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_testLSTM, y_testLSTM = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)), np.array(y_test)

    y_hatLSTM = model.predict(X_testLSTM)

    y_hat_rbf = regr_rbf.predict(X_test)
    y_hat_poly = regr_poly.predict(X_test)
    y_hat_lin = regr_lin.predict(X_test)
    y_hat_rfr = regr_rfr.predict(X_test)
    y_hat_gbr = regr_gbr.predict(X_test)

    y_hatLSTM = scalerY.inverse_transform(y_hatLSTM)
    y_testLSTM = scalerY.inverse_transform(y_testLSTM)

    y_hat_poly = np.reshape(y_hat_poly, (y_hat_poly.shape[0], 1))
    y_hat_rbf = np.reshape(y_hat_rbf, (y_hat_rbf.shape[0], 1))
    y_hat_lin = np.reshape(y_hat_lin, (y_hat_lin.shape[0], 1))
    y_hat_rfr = np.reshape(y_hat_rfr, (y_hat_rfr.shape[0], 1))
    y_hat_gbr = np.reshape(y_hat_gbr, (y_hat_gbr.shape[0], 1))


    y_hat_poly = scalerY.inverse_transform(y_hat_poly)
    y_hat_rbf = scalerY.inverse_transform(y_hat_rbf)
    y_hat_lin = scalerY.inverse_transform(y_hat_lin)
    y_hat_rfr = scalerY.inverse_transform(y_hat_rfr)
    y_hat_gbr = scalerY.inverse_transform(y_hat_gbr)
    y_test = scalerY.inverse_transform(y_test)
    
    
    y_test = [float(i) for i in np.reshape(y_test, (y_test.shape[0]))]
    y_testLSTM = [float(i) for i in np.reshape(y_testLSTM, (y_testLSTM.shape[0]))]
    y_hatLSTM = [float(i) for i in np.reshape(y_hatLSTM, (y_hatLSTM.shape[0]))]
    y_hat_lin = [float(i) for i in np.reshape(y_hat_lin, (y_hat_lin.shape[0]))]
    y_hat_poly = [float(i) for i in np.reshape(y_hat_poly, (y_hat_poly.shape[0]))]
    y_hat_rbf = [float(i) for i in np.reshape(y_hat_rbf, (y_hat_rbf.shape[0]))]
    y_hat_rfr = [float(i) for i in np.reshape(y_hat_rfr, (y_hat_rfr.shape[0]))]
    y_hat_gbr = [float(i) for i in np.reshape(y_hat_gbr, (y_hat_gbr.shape[0]))]
    # print(y_test.shape, y_testLSTM.shape, y_hatLSTM.shape, y_hat_gbr.shape)

    test_wdate = df_copy['Date'][-test_size - 1:]
    test_wdate = pd.to_datetime(test_wdate, infer_datetime_format=True)
    test_wdate = pd.DataFrame(data= test_wdate.tolist(), columns=['Date'], index=list(range(test_size + 1)))



    lstm_pred = pd.DataFrame(data= y_hatLSTM, columns=['Adj Close'])
    lstm_pred = pd.merge(lstm_pred, test_wdate, left_index=True, right_index=True, how='outer')

    svrLin_pred = pd.DataFrame(data= y_hat_lin, columns=['Adj Close'])
    svrLin_pred = pd.merge(svrLin_pred, test_wdate, left_index=True, right_index=True, how='outer')

    svrPoly_pred = pd.DataFrame(data= y_hat_poly, columns=['Adj Close'])
    svrPoly_pred = pd.merge(svrPoly_pred, test_wdate, left_index=True, right_index=True, how='outer')

    svrRbf_pred = pd.DataFrame(data= y_hat_rbf, columns=['Adj Close'])
    svrRbf_pred = pd.merge(svrRbf_pred, test_wdate, left_index=True, right_index=True, how='outer')

    rfr_pred = pd.DataFrame(data= y_hat_rfr, columns=['Adj Close'])
    rfr_pred = pd.merge(rfr_pred, test_wdate, left_index=True, right_index=True, how='outer')

    gbr_pred = pd.DataFrame(data= y_hat_gbr, columns=['Adj Close'])
    gbr_pred = pd.merge(gbr_pred, test_wdate, left_index=True, right_index=True, how='outer')

    os.chdir(path_agent_data)

    lstm_pred.to_csv(stock_name+'_LSTM.csv')
    svrLin_pred.to_csv(stock_name+'_SVR_Lin.csv')
    svrPoly_pred.to_csv(stock_name+'_SVR_Poly.csv')
    svrRbf_pred.to_csv(stock_name+'_SVR_Rbf.csv')
    rfr_pred.to_csv(stock_name+'_RFR.csv')
    gbr_pred.to_csv(stock_name+'_GBR.csv')


    return y_test, y_testLSTM, y_hatLSTM[1:], y_hat_lin[1:], y_hat_poly[1:], y_hat_rbf[1:], y_hat_rfr[1:], y_hat_gbr[1:]

    # MSE_rbf = mean_squared_error(y_hat_rbf, y_test)
    # MSE_poly = mean_squared_error(y_hat_poly, y_test)
    # MSE_lin = mean_squared_error(y_hat_lin, y_test)
    # MSE_rfr = mean_squared_error(y_hat_rfr, y_test)
    # MSE_gbr = mean_squared_error(y_hat_gbr, y_test)
    # MSE_LSTM = mean_squared_error(y_hatLSTM, y_testLSTM)

    # MAE_rbf = mean_absolute_error(y_hat_rbf, y_test)
    # MAE_poly = mean_absolute_error(y_hat_poly, y_test)
    # MAE_lin = mean_absolute_error(y_hat_lin, y_test)
    # MAE_rfr = mean_absolute_error(y_hat_rfr, y_test)
    # MAE_gbr = mean_absolute_error(y_hat_gbr, y_test)
    # MAE_LSTM = mean_absolute_error(y_hatLSTM, y_testLSTM)

    # DA_LSTM = directional_asymmetry(y_hatLSTM, y_testLSTM)
    # DA_lin = directional_asymmetry(y_hat_lin, y_test)
    # DA_poly = directional_asymmetry(y_hat_poly, y_test)
    # DA_rbf = directional_asymmetry(y_hat_rbf, y_test)
    # DA_rfr = directional_asymmetry(y_hat_rfr, y_test)
    # DA_gbr = directional_asymmetry(y_hat_gbr, y_test)

    # mse = minmax_scale(np.transpose([MSE_LSTM, MSE_lin, MSE_poly, MSE_rbf, MSE_rfr, MSE_gbr]))
    # mae = minmax_scale(np.transpose([MAE_LSTM, MAE_lin, MAE_poly, MAE_rbf, MAE_rfr, MAE_gbr]))
    # da = [DA_LSTM, DA_lin, DA_poly, DA_rbf, DA_rfr, DA_gbr]


    # # LSTM
    # stock_close_LSTM = stock_close


    # # MACD Values
    # #Compute MACD and MACD Histogram
    # for i in range(15):
    #     macdata_LSTM, machdata_LSTM = getMACD_MHIST(stock_close_LSTM)
    #     rsidata_LSTM = getRSI(stock_close_LSTM)
    #     bbdata_LSTM = getBB(stock_close_LSTM)


    #     Close = pd.DataFrame({'Close': stock_close_LSTM})
    #     RSI = pd.DataFrame({'RSI': rsidata_LSTM['RSI']})
    #     BB = pd.DataFrame({'BB': bbdata_LSTM['Strategy']})
    #     MACD = pd.DataFrame({'MACD': macdata_LSTM['MACD']})
    #     MHIST = pd.DataFrame({'MHIST': machdata_LSTM['MHIST']})
    #     Middle = pd.DataFrame({'Middle' : bbdata_LSTM['Middle']})
    #     Upper = pd.DataFrame({'Upper' : bbdata_LSTM['Upper']})
    #     Lower = pd.DataFrame({'Lower' : bbdata_LSTM['Lower']})

    #     # Merging into Single DataFrame

    #     merge1 = pd.merge(Close, RSI, left_index=True, right_index=True, how='outer')
    #     merge2 = pd.merge(merge1, MACD, left_index=True, right_index=True, how='outer')
    #     merge3 = pd.merge(merge2, MHIST, left_index=True, right_index=True, how='outer')
    #     merge4 = pd.merge(merge3, Middle, left_index=True, right_index=True, how='outer')
    #     merge5 = pd.merge(merge4, Upper, left_index=True, right_index=True, how='outer')
    #     df_finalLSTM = pd.merge(merge5, Lower, left_index=True, right_index=True, how='outer')
    #     df_finalLSTM.dropna(inplace=True)


    #     test_data_LSTM = df_finalLSTM
    #     X_test_LSTM = test_data_LSTM[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]

    #     X_test_LSTM = scalerX.fit_transform(X_test_LSTM)

    #     X_test_LSTM = np.array(X_test_LSTM)
    #     X_testLSTM2 = np.reshape(X_test_LSTM, (X_test_LSTM.shape[0], X_test_LSTM.shape[1], 1))


    #     X_testLSTM2 = X_testLSTM2[-1]
    #     X_testLSTM2 = np.reshape(X_testLSTM2, (1, X_testLSTM2.shape[0], X_testLSTM2.shape[1]))


    #     y_hatLSTM2 = model.predict(X_testLSTM2)
    #     y_hatLSTM2 = scalerY.inverse_transform(y_hatLSTM2)

    #     stock_close_LSTM.loc[stock_close_LSTM.index.max() + 1] = y_hatLSTM2[-1][0]
    #     y_hatLSTM = np.append(y_hatLSTM, [[y_hatLSTM2[-1][0]]], axis=0)



    # # SVR-Linear
    # stock_close_SVR_Lin = stock_close

    
    # # MACD Values
    # #Compute MACD and MACD Histogram
    # for i in range(15):
    #     macdata_SVR_Lin, machdata_SVR_Lin = getMACD_MHIST(stock_close_SVR_Lin)
    #     rsidata_SVR_Lin = getRSI(stock_close_SVR_Lin)
    #     bbdata_SVR_Lin = getBB(stock_close_SVR_Lin)


    #     Close = pd.DataFrame({'Close': stock_close_SVR_Lin})
    #     RSI = pd.DataFrame({'RSI': rsidata_SVR_Lin['RSI']})
    #     BB = pd.DataFrame({'BB': bbdata_SVR_Lin['Strategy']})
    #     MACD = pd.DataFrame({'MACD': macdata_SVR_Lin['MACD']})
    #     MHIST = pd.DataFrame({'MHIST': machdata_SVR_Lin['MHIST']})
    #     Middle = pd.DataFrame({'Middle' : bbdata_SVR_Lin['Middle']})
    #     Upper = pd.DataFrame({'Upper' : bbdata_SVR_Lin['Upper']})
    #     Lower = pd.DataFrame({'Lower' : bbdata_SVR_Lin['Lower']})

    #     # Merging into Single DataFrame

    #     merge1 = pd.merge(Close, RSI, left_index=True, right_index=True, how='outer')
    #     merge2 = pd.merge(merge1, MACD, left_index=True, right_index=True, how='outer')
    #     merge3 = pd.merge(merge2, MHIST, left_index=True, right_index=True, how='outer')
    #     merge4 = pd.merge(merge3, Middle, left_index=True, right_index=True, how='outer')
    #     merge5 = pd.merge(merge4, Upper, left_index=True, right_index=True, how='outer')
    #     df_finalSVR_Lin = pd.merge(merge5, Lower, left_index=True, right_index=True, how='outer')
    #     df_finalSVR_Lin.dropna(inplace=True)




    #     test_data_SVR_Lin = df_finalSVR_Lin
    #     X_test_SVR_Lin = test_data_SVR_Lin[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]




    #     X_test_SVR_Lin = scalerX.fit_transform(X_test_SVR_Lin)

    #     X_test_SVR_Lin = np.array(X_test_SVR_Lin)

    #     X_test_SVR_Lin = X_test_SVR_Lin[-1]
    #     X_test_SVR_Lin = np.reshape(X_test_SVR_Lin, (1, X_test_SVR_Lin.shape[0]))


    #     y_hatSVR_Lin = regr_lin.predict(X_test_SVR_Lin)

    #     y_hatSVR_Lin = np.reshape(y_hatSVR_Lin, (y_hatSVR_Lin.shape[0], 1))

    #     y_hatSVR_Lin = scalerY.inverse_transform(y_hatSVR_Lin)



    #     stock_close_SVR_Lin.loc[stock_close_SVR_Lin.index.max() + 1] = y_hatSVR_Lin[-1][0]
    #     y_hat_lin = np.append(y_hat_lin, [[y_hatSVR_Lin[-1][0]]], axis=0)

    # # SVR-Polynomial
    # stock_close_SVR_Poly = stock_close

    
    # # MACD Values
    # #Compute MACD and MACD Histogram
    # for i in range(15):
    #     macdata_SVR_Poly, machdata_SVR_Poly = getMACD_MHIST(stock_close_SVR_Poly)
    #     rsidata_SVR_Poly = getRSI(stock_close_SVR_Poly)
    #     bbdata_SVR_Poly = getBB(stock_close_SVR_Poly)


    #     Close = pd.DataFrame({'Close': stock_close_SVR_Poly})
    #     RSI = pd.DataFrame({'RSI': rsidata_SVR_Poly['RSI']})
    #     BB = pd.DataFrame({'BB': bbdata_SVR_Poly['Strategy']})
    #     MACD = pd.DataFrame({'MACD': macdata_SVR_Poly['MACD']})
    #     MHIST = pd.DataFrame({'MHIST': machdata_SVR_Poly['MHIST']})
    #     Middle = pd.DataFrame({'Middle' : bbdata_SVR_Poly['Middle']})
    #     Upper = pd.DataFrame({'Upper' : bbdata_SVR_Poly['Upper']})
    #     Lower = pd.DataFrame({'Lower' : bbdata_SVR_Poly['Lower']})

    #     # Merging into Single DataFrame

    #     merge1 = pd.merge(Close, RSI, left_index=True, right_index=True, how='outer')
    #     merge2 = pd.merge(merge1, MACD, left_index=True, right_index=True, how='outer')
    #     merge3 = pd.merge(merge2, MHIST, left_index=True, right_index=True, how='outer')
    #     merge4 = pd.merge(merge3, Middle, left_index=True, right_index=True, how='outer')
    #     merge5 = pd.merge(merge4, Upper, left_index=True, right_index=True, how='outer')
    #     df_finalSVR_Poly = pd.merge(merge5, Lower, left_index=True, right_index=True, how='outer')
    #     df_finalSVR_Poly.dropna(inplace=True)




    #     test_data_SVR_Poly = df_finalSVR_Poly
    #     X_test_SVR_Poly = test_data_SVR_Poly[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]




    #     X_test_SVR_Poly = scalerX.fit_transform(X_test_SVR_Poly)

    #     X_test_SVR_Poly = np.array(X_test_SVR_Poly)

    #     X_test_SVR_Poly = X_test_SVR_Poly[-1]
    #     X_test_SVR_Poly = np.reshape(X_test_SVR_Poly, (1, X_test_SVR_Poly.shape[0]))


    #     y_hatSVR_Poly = regr_poly.predict(X_test_SVR_Poly)

    #     y_hatSVR_Poly = np.reshape(y_hatSVR_Poly, (y_hatSVR_Poly.shape[0], 1))

    #     y_hatSVR_Poly = scalerY.inverse_transform(y_hatSVR_Poly)


    #     stock_close_SVR_Poly.loc[stock_close_SVR_Poly.index.max() + 1] = y_hatSVR_Poly[-1][0]
    #     y_hat_poly = np.append(y_hat_poly, [[y_hatSVR_Poly[-1][0]]], axis=0)


    # # SVR-Rbf
    # stock_close_SVR_Rbf = stock_close

    
    # # MACD Values
    # #Compute MACD and MACD Histogram
    # for i in range(15):
    #     macdata_SVR_Rbf, machdata_SVR_Rbf = getMACD_MHIST(stock_close_SVR_Rbf)
    #     rsidata_SVR_Rbf = getRSI(stock_close_SVR_Rbf)
    #     bbdata_SVR_Rbf = getBB(stock_close_SVR_Rbf)


    #     Close = pd.DataFrame({'Close': stock_close_SVR_Rbf})
    #     RSI = pd.DataFrame({'RSI': rsidata_SVR_Rbf['RSI']})
    #     BB = pd.DataFrame({'BB': bbdata_SVR_Rbf['Strategy']})
    #     MACD = pd.DataFrame({'MACD': macdata_SVR_Rbf['MACD']})
    #     MHIST = pd.DataFrame({'MHIST': machdata_SVR_Rbf['MHIST']})
    #     Middle = pd.DataFrame({'Middle' : bbdata_SVR_Rbf['Middle']})
    #     Upper = pd.DataFrame({'Upper' : bbdata_SVR_Rbf['Upper']})
    #     Lower = pd.DataFrame({'Lower' : bbdata_SVR_Rbf['Lower']})

    #     # Merging into Single DataFrame

    #     merge1 = pd.merge(Close, RSI, left_index=True, right_index=True, how='outer')
    #     merge2 = pd.merge(merge1, MACD, left_index=True, right_index=True, how='outer')
    #     merge3 = pd.merge(merge2, MHIST, left_index=True, right_index=True, how='outer')
    #     merge4 = pd.merge(merge3, Middle, left_index=True, right_index=True, how='outer')
    #     merge5 = pd.merge(merge4, Upper, left_index=True, right_index=True, how='outer')
    #     df_finalSVR_Rbf = pd.merge(merge5, Lower, left_index=True, right_index=True, how='outer')
    #     df_finalSVR_Rbf.dropna(inplace=True)




    #     test_data_SVR_Rbf = df_finalSVR_Rbf
    #     X_test_SVR_Rbf = test_data_SVR_Rbf[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]




    #     X_test_SVR_Rbf = scalerX.fit_transform(X_test_SVR_Rbf)

    #     X_test_SVR_Rbf = np.array(X_test_SVR_Rbf)

    #     X_test_SVR_Rbf = X_test_SVR_Rbf[-1]
    #     X_test_SVR_Rbf = np.reshape(X_test_SVR_Rbf, (1, X_test_SVR_Rbf.shape[0]))


    #     y_hatSVR_Rbf = regr_rbf.predict(X_test_SVR_Rbf)

    #     y_hatSVR_Rbf = np.reshape(y_hatSVR_Rbf, (y_hatSVR_Rbf.shape[0], 1))

    #     y_hatSVR_Rbf = scalerY.inverse_transform(y_hatSVR_Rbf)



    #     stock_close_SVR_Rbf.loc[stock_close_SVR_Rbf.index.max() + 1] = y_hatSVR_Rbf[-1][0]
    #     y_hat_rbf = np.append(y_hat_rbf, [[y_hatSVR_Rbf[-1][0]]], axis=0)


    # # RFR
    # stock_close_RFR = stock_close

    
    # # MACD Values
    # #Compute MACD and MACD Histogram
    # for i in range(15):
    #     macdata_RFR, machdata_RFR = getMACD_MHIST(stock_close_RFR)
    #     rsidata_RFR = getRSI(stock_close_RFR)
    #     bbdata_RFR = getBB(stock_close_RFR)


    #     Close = pd.DataFrame({'Close': stock_close_RFR})
    #     RSI = pd.DataFrame({'RSI': rsidata_RFR['RSI']})
    #     BB = pd.DataFrame({'BB': bbdata_RFR['Strategy']})
    #     MACD = pd.DataFrame({'MACD': macdata_RFR['MACD']})
    #     MHIST = pd.DataFrame({'MHIST': machdata_RFR['MHIST']})
    #     Middle = pd.DataFrame({'Middle' : bbdata_RFR['Middle']})
    #     Upper = pd.DataFrame({'Upper' : bbdata_RFR['Upper']})
    #     Lower = pd.DataFrame({'Lower' : bbdata_RFR['Lower']})

    #     # Merging into Single DataFrame

    #     merge1 = pd.merge(Close, RSI, left_index=True, right_index=True, how='outer')
    #     merge2 = pd.merge(merge1, MACD, left_index=True, right_index=True, how='outer')
    #     merge3 = pd.merge(merge2, MHIST, left_index=True, right_index=True, how='outer')
    #     merge4 = pd.merge(merge3, Middle, left_index=True, right_index=True, how='outer')
    #     merge5 = pd.merge(merge4, Upper, left_index=True, right_index=True, how='outer')
    #     df_finalRFR = pd.merge(merge5, Lower, left_index=True, right_index=True, how='outer')
    #     df_finalRFR.dropna(inplace=True)




    #     test_data_RFR = df_finalRFR
    #     X_test_RFR = test_data_RFR[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]




    #     X_test_RFR = scalerX.fit_transform(X_test_RFR)

    #     X_test_RFR = np.array(X_test_RFR)

    #     X_test_RFR = X_test_RFR[-1]
    #     X_test_RFR = np.reshape(X_test_RFR, (1, X_test_RFR.shape[0]))


    #     y_hatRFR = regr_rfr.predict(X_test_RFR)

    #     y_hatRFR = np.reshape(y_hatRFR, (y_hatRFR.shape[0], 1))

    #     y_hatRFR = scalerY.inverse_transform(y_hatRFR)



    #     stock_close_RFR.loc[stock_close_RFR.index.max() + 1] = y_hatRFR[-1][0]
    #     y_hat_rfr = np.append(y_hat_rfr, [[y_hatRFR[-1][0]]], axis=0)

    # # GBR
    # stock_close_GBR = stock_close

    
    # # MACD Values
    # #Compute MACD and MACD Histogram
    # for i in range(15):
    #     macdata_GBR, machdata_GBR = getMACD_MHIST(stock_close_GBR)
    #     rsidata_GBR = getRSI(stock_close_GBR)
    #     bbdata_GBR = getBB(stock_close_GBR)


    #     Close = pd.DataFrame({'Close': stock_close_GBR})
    #     RSI = pd.DataFrame({'RSI': rsidata_GBR['RSI']})
    #     BB = pd.DataFrame({'BB': bbdata_GBR['Strategy']})
    #     MACD = pd.DataFrame({'MACD': macdata_GBR['MACD']})
    #     MHIST = pd.DataFrame({'MHIST': machdata_GBR['MHIST']})
    #     Middle = pd.DataFrame({'Middle' : bbdata_GBR['Middle']})
    #     Upper = pd.DataFrame({'Upper' : bbdata_GBR['Upper']})
    #     Lower = pd.DataFrame({'Lower' : bbdata_GBR['Lower']})

    #     # Merging into Single DataFrame

    #     merge1 = pd.merge(Close, RSI, left_index=True, right_index=True, how='outer')
    #     merge2 = pd.merge(merge1, MACD, left_index=True, right_index=True, how='outer')
    #     merge3 = pd.merge(merge2, MHIST, left_index=True, right_index=True, how='outer')
    #     merge4 = pd.merge(merge3, Middle, left_index=True, right_index=True, how='outer')
    #     merge5 = pd.merge(merge4, Upper, left_index=True, right_index=True, how='outer')
    #     df_finalGBR = pd.merge(merge5, Lower, left_index=True, right_index=True, how='outer')
    #     df_finalGBR.dropna(inplace=True)




    #     test_data_GBR = df_finalGBR
    #     X_test_GBR = test_data_GBR[['Close', 'MACD', 'MHIST', 'RSI', 'Middle', 'Upper', 'Lower']]


    #     X_test_GBR = scalerX.fit_transform(X_test_GBR)

    #     X_test_GBR = np.array(X_test_GBR)

    #     X_test_GBR = X_test_GBR[-1]
    #     X_test_GBR = np.reshape(X_test_GBR, (1, X_test_GBR.shape[0]))


    #     y_hatGBR = regr_gbr.predict(X_test_GBR)

    #     y_hatGBR = np.reshape(y_hatGBR, (y_hatGBR.shape[0], 1))

    #     y_hatGBR = scalerY.inverse_transform(y_hatGBR)



    #     stock_close_GBR.loc[stock_close_GBR.index.max() + 1] = y_hatGBR[-1][0]
    #     y_hat_gbr = np.append(y_hat_gbr, [[y_hatGBR[-1][0]]], axis=0)


    # return y_test, y_testLSTM, y_hatLSTM[1:], y_hat_lin[1:], y_hat_poly[1:], y_hat_rbf[1:], y_hat_rfr[1:], y_hat_gbr[1:]