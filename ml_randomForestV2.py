import csv
import pandas as pd  
import matplotlib.pyplot as plt
import time
import sys
import os
import numpy as np
from binance.client import Client
from binance.enums import *
from binance_api_keys import *
current_symbol = "LTCETH"
kline_min = 15
kline_interval = "15m"
#UPDATE_DATA = True
UPDATE_DATA = False
MAKE_NEW_DATASET = False
# MAKE_NEW_DATASET = True
if len(sys.argv) > 1:
    if sys.argv[1]:
        current_symbol = str(sys.argv[1])
        
    if sys.argv[2]:
        kline_interval = str(sys.argv[2])+"m"
        kline_min = int(sys.argv[2])
    if len(sys.argv) > 3:
        if sys.argv[3] == "-n":
            MAKE_NEW_DATASET = True
query_limit=1000        
time_stepper = kline_min*60*query_limit*1000

print("Current Symbol:", current_symbol)
print("Kline interval", kline_interval)

def get_history_data(current_symbol):
    print("Updating kline history data...", end=' ')
    current_millis = int(round(time.time() * 1000))
    
    df=pd.DataFrame()
    for i in range(10,0,-1):
        TimeMillis = current_millis-(i*time_stepper)
        df1 = client.get_klines(symbol=current_symbol, interval=kline_interval, startTime=TimeMillis,limit=query_limit)
        df1 = pd.DataFrame(df1, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume',' Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Can be ignored'])
        df1.set_index('Time')
        
   
        df = df.append(df1, ignore_index=True)
        time.sleep(0.03)
    df["Time"] = pd.to_datetime(df["Time"],unit="ms")
    # print df[['Time','Close']]
    # return df[['Time','Close']]
    # data =  df[['Time','Close']]
    data = df
    data.to_csv('data/'+current_symbol + kline_interval+'.csv', index=False)
    print("[DONE]")
def get_data(current_symbol):
    df = pd.read_csv("data/"+current_symbol + kline_interval+".csv", usecols=['Time', 'Open', 'High', 'Low', 'Close', 'Volume',' Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume'])
    return df
def get_live_data(current_symbol):
    df=pd.DataFrame() 
    current_millis = int(round(time.time() * 1000))    
    df1 = client.get_klines(symbol=current_symbol, interval=kline_interval, endTime=current_millis,limit=16)
    df1 = pd.DataFrame(df1, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume',' Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Can be ignored'])
    df1.set_index('Time')
    df = df.append(df1, ignore_index=True)
    df["Time"] = pd.to_datetime(df["Time"],unit="ms")
    return df
def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # TODO: Compute and return rolling standard deviation
    return values.rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # TODO: Compute upper_band and lower_band
    upper_band = rm+2*rstd
    lower_band = rm-2*rstd
    return upper_band, lower_band
if UPDATE_DATA:

    get_history_data(current_symbol)
else:
    print("Old kline history data used")
df = get_data(current_symbol)
# print df
rm = get_rolling_mean(df["Close"], window=20)
rstd = get_rolling_std(df["Close"], window=20)
upper_band, lower_band = get_bollinger_bands(rm, rstd)

data_x = pd.DataFrame()
# data = data.append(rm)
# data = data.append(rstd)
df_lenght = len(df)
print("Datapoints in history data: ", df_lenght)
index_array= []
for i in range(300):
    index_array.append(i)
# print index_array


# print data
period_multiplier=16
future_period_tester=9
colums_amount = 9
row_temp=[]
data_temp_x=np.empty((0, colums_amount*period_multiplier+1))
data_temp_y=np.empty((0, 1))
# print data_temp_y


if MAKE_NEW_DATASET:
    print("Making new dataset...")
    for index in range(0, df_lenght-period_multiplier-future_period_tester):
        
        # if index == 10:
            # break
        for row in df[index:(index+period_multiplier)].itertuples(): # puts period_multiplier amount of past data for our X row value to use for predictions
        
            # print(row)
            # print(getattr(row, 'Index'), getattr(row, 'Open'))
            row_temp.append(getattr(row, 'Open'))
            row_temp.append(getattr(row, 'High'))
            row_temp.append(getattr(row, 'Low'))
            row_temp.append(getattr(row, 'Close'))
            row_temp.append(getattr(row, 'Volume'))
            row_temp.append(getattr(row, '_8'))
            row_temp.append(getattr(row, '_9'))
            row_temp.append(getattr(row, '_10'))
            row_temp.append(getattr(row, '_11'))
        # for i in range(0, period_multiplier):
                
            # row_temp.append(df.loc[(index+i),'Close'])
            # row_temp.append(df.loc[(index+i),'High'])
            # row_temp.append(df.loc[(index+i),'Low'])
            # row_temp.append(df.loc[(index+i),'Close'])
            # row_temp.append(df.loc[(index+i),'Volume'])
            # row_temp.append(df.loc[(index+i),'Quote asset volume'])
            # row_temp.append(df.loc[(index+i),'Number of trades'])
            # row_temp.append(df.loc[(index+i),'Taker buy base asset volume'])
            # row_temp.append(df.loc[(index+i),'Taker buy quote asset volume'])
        
        
        
        for i in range(1,future_period_tester): # lets check where the price goes in the future to get our Y value
            future_close = df.loc[(index+period_multiplier+i),'Close']
            # print "number",index+period_multiplier+i," /"
            # print df.loc[(index+period_multiplier),'Close']
            # print future_close /df.loc[(index+period_multiplier),'Close']
            
            if(future_close /df.loc[(index+period_multiplier),'Close']) > 1.01:
                future_class = [2]
                break
            
            elif (future_close /df.loc[(index+period_multiplier),'Close']) < 0.99:
            
                future_class = [0]
                break
            
            else:
                future_class = [1]
       
        row_temp.append(future_class[0])
        data_temp_x = np.append(data_temp_x, [row_temp], axis=0)
        data_temp_y = np.append(data_temp_y, [future_class], axis=0)
        
      
        row_temp=[]
       
        # sys.stdout.write("\r" + str(index))
        # sys.stdout.flush()
        print(str(index),end='\r')
        # if index == 3000:
            # break
    # print len(row_temp) 
    # print len(index_array) 
    # data.append(row_temp)
    # exit()
    data_x = pd.DataFrame(data_temp_x)
    data_y = pd.DataFrame(data_temp_y)
    # y_target= data_temp_y.flatten().astype('float')
   

    #Add data to file for next time
    data_x.to_csv('data/'+current_symbol + kline_interval+'_x_data.csv', index=False)
    data_y.to_csv('data/'+current_symbol + kline_interval+'_y_data.csv', index=False)
    print(pd.value_counts(data_y[0]))

else:
    print("Old dataset data loaded")
    data_x = pd.read_csv("data/"+current_symbol + kline_interval+"_x_data.csv")
    data_y = pd.read_csv("data/"+current_symbol + kline_interval+"_y_data.csv")
    print(pd.value_counts(data_y['0']))
# print df.loc[(200),'Close']

# print data_x.tail
# print data_temp_y.shape
y_target= data_y.values.flatten()
# data = pd.DataFrame(row_temp)
# print data_y
# print y_target
# for i in range(0, len(y_target)):

print("Test set contains following classes:")

# print data_y.count(1)
# print data_y.count(0)
# print data_y.count(-1)
# print data_y.count(-2)

    # print y_target[i]
# print X_test.shape
# exit()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
X_train, X_test, y_train, y_test = train_test_split( data_x, y_target, shuffle=False)
     
# if os.path.isfile('forest.joblib'):
#     print("Classifier loaded from file")
#     forest = joblib.load('forest.joblib')
# else:
#     print("Classifier recalculated")
# forest = RandomForestClassifier(n_estimators=700, max_depth=3, max_features=20,n_jobs=-1)
# forest.fit(X_train, y_train)
    #save to disk
    # joblib.dump(forest, 'forest.joblib')

"""Lets test hyperparameters in a loop"""
best_acc= 0.0
best_n_features = 2
# for x in range(1,4):
#     # max_depth= x*2+1
#     max_features= x
#     forest = RandomForestClassifier(n_estimators=600, max_depth=17, max_features=max_features,n_jobs=-1)
#     forest.fit(X_train, y_train)
#     test_score = forest.score(X_test, y_test)
#     print("max_features:",max_features)
#     print(("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train))))
#     print(("Accuracy on test set: {:.3f}".format(test_score)))
#     if test_score > best_acc:
#         best_n_features = max_features
print("best max features: "+ str(best_n_features))
forest = RandomForestClassifier(n_estimators=600, max_depth=17, max_features=1,n_jobs=-1)
forest.fit(X_train, y_train)
results = forest.predict(X_test)
# print results[:200]
# print y_test
# print X_test

test_buy=0
hit=0
miss=0
bhit=0
bmiss=0
hit_1=0
hit_0=0
hit_1neg=0
hit_2=0
hit_2neg=0

# Test how did we do on predictions
for i in range(0,len(results)):
    # for x in range(-2,2,1):
        # if x == 0:
           # continue
    if results[i] == 1:
        if y_test[i]==1:
            hit_1=hit_1+1
        if y_test[i]==2:
            hit_2=hit_2+1
        if y_test[i]==0:
            hit_0=hit_0+1
        if y_test[i]==-1:
            hit_1neg=hit_1neg+1
        if y_test[i]==-2:
            hit_2neg=hit_2neg+1
        test_buy=test_buy+y_test[i]
        
# if results[i] ==2:
        # print results[i] , " : " , y_test[i]
        # if results[i] == y_test[i]:
            # hit=hit+1
        # else:
            # miss=miss+1
    if  y_test[i]==1:
       
        if results[i] == y_test[i]:
            bhit=bhit+1
        else:
            bmiss=bmiss+1
    
# print "Hit :",hit_1,"miss:",hit_1
total_hits= hit_2neg+hit_2+hit_1neg+hit_1+hit_0
print("When test data was 1, predict got it a",bhit,"hits and and",bmiss,"miss")
print(("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train))))
print(("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test))))
print("When predicting buy signal [1], real results are:")
print("2:",hit_2)
print("1:",hit_1)
print("0:",hit_0)
print("-1:",hit_1neg)
print("-2:",hit_2neg)
print("Odds for over 1:",(hit_1+hit_2)/total_hits*100,"odds for zero: ",hit_0/total_hits*100,"odds for -1", (hit_1neg+hit_2neg)/total_hits*100 )
print("Tradebot profit:", test_buy)
# exit()
# x_new = np.array(X_test.iloc[8]).reshape(1,-1)
# print x_new
# print forest.predict(x_new)
# print y_test[8]

print("Starting live test...")
new_data = get_live_data(current_symbol)
# print new_data['Time'].tail
row_temp_new=[]
for row in new_data.itertuples():
        
            # print(row)
            # print(getattr(row, 'Index'), getattr(row, 'Open'))
            row_temp_new.append(getattr(row, 'Open'))
            row_temp_new.append(getattr(row, 'High'))
            row_temp_new.append(getattr(row, 'Low'))
            row_temp_new.append(getattr(row, 'Close'))
            row_temp_new.append(getattr(row, 'Volume'))
            row_temp_new.append(getattr(row, '_8'))
            row_temp_new.append(getattr(row, '_9'))
            row_temp_new.append(getattr(row, '_10'))
            row_temp_new.append(getattr(row, '_11'))

new_data_x = np.array(row_temp_new).reshape(1,-1)
# print new_data_x.shape
predict_period = (future_period_tester-1)*int(kline_interval[:-1])
print("In the next ",predict_period,"minutes the price predict class is :",forest.predict(new_data_x))

