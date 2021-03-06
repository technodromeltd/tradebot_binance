#!/usr/bin/env python
# coding=utf-8
"""
***USE AT YOUR OWN RISK***
Trading bot using Binance API to make real trades based on Bollinger bands analysis.
Can only trade between two assets using limit bids, where the sell price need to be higher than current price,  
so in theory cant lose all of your money by mistakenly selling too low :)
You need to set up binance api account first and add your key and secret to binance_api_keys.py file
Usage:
    main.py <COINPAIR> <UPDATE_INTERVAL_S>
Arguments:
    <COINPAIR>          : e.g. "LTCETH" (default)
    <UPDATE_INTERVAL_S> : e.g. 15 (default)
Other Settings:
    upper_band_m        : Selling when price is higher than upper_band_m * bollinger bands upper value (current_price > upper_band_m * upper_band) *This value is based on backprogatation tests on past data  
    lower_band_m        : Buying when (current_price < lower_band_m * lower_band) *This value is based on backprogatation tests on past data  
    data_limit          : How many past kline price data intervals to get
    data_kline_interval : What kline price data interval to use, KLINE_INTERVAL_1MINUTE to 1 KLINE_INTERVAL_1MONTH 
        
"""
from binance_api_keys import *
import csv
import datetime
import pandas as pd  
import datetime
import time
from binance.client import Client
from binance.enums import *
import math
import sys
import colorama
from colorama import Fore, init
client = Client(api_key, secret_key)
starting_time = int(round(time.time()))

current_symbol = "LTCETH" # Symbol pair to trade
update_interval = 15  # Seconds. How often to check prices
# Read Arguments
if len(sys.argv) > 1:
    if sys.argv[1]:
        current_symbol = str(sys.argv[1])
    if len(sys.argv) > 2:
        if sys.argv[2]:
            update_interval = float(sys.argv[2])
if len(current_symbol) > 6:
    
    coin_b = current_symbol[4:]   
    coin_a = current_symbol[:4]    
else:    
    coin_a = current_symbol[:3]    
    coin_b = current_symbol[3:]

class TradeBot:
    """
    TODO Refactor rest of functions inside class
    """

    #SETTINGS
    data_last_updated = 0
   
    data_kline_interval = KLINE_INTERVAL_3MINUTE # kline price data interval
   
    data_limit = 20 #
    upper_band_m = 1.02 # Upper bollinger band * this to   to sell
    lower_band_m = 0.99 # Lowe bolling band limit to buy
    max_single_buy_eth = 1.0


    def __init__(self, current_symbol=None,update_interval=None):
        self.current_symbol = current_symbol
        self.update_interval = update_interval



    def get_new_data(self):

        current_millis = int(round(time.time() * 1000))
        df=pd.DataFrame()
        
        df1 = client.get_klines(symbol=self.current_symbol, interval=self.data_kline_interval, endTime=current_millis, limit=self.data_limit)
        df1 = pd.DataFrame(df1, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume',' Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Can be ignored'])
        df1.set_index('Time')
    
        df = df.append(df1, ignore_index=True)
        df["Time"] = pd.to_datetime(df["Time"],unit="ms")

        data =  df[['Time','Close']]
        return data

    
    def get_latest_price(self):
        current_millis = int(round(time.time() * 1000))
        df1 = client.get_klines(symbol=self.current_symbol, interval=KLINE_INTERVAL_1MINUTE, endTime=current_millis, limit=1)
        df1 = pd.DataFrame(df1, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume',' Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Can be ignored'])
        
        return df1['Close'][0]

    def get_data(self):
        df_temp = pd.read_csv("data/"+this.current_symbol+".csv", usecols=["Time","Close"])
        return df_temp

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()
def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()
def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm+2*rstd
    lower_band = rm-2*rstd
    return upper_band, lower_band
    
def compute_daily_returns(df):
    """Compute and return the daily return values."""
   
    df = df.pct_change(1)
    df.fillna(0, inplace=True)

    return df

def make_order(type, order_price):
    global min_sell_amount
    if (type=='buy'):
        #Trying to buy
        
        balance = client.get_asset_balance(asset=coin_b)
        amount  = bot.max_single_buy_eth / order_price # Buy only with 1eth at once
        global balance_coin_b
        balance_coin_b = float(balance['free'])
        if balance_coin_b > bot.max_single_buy_eth:
            # Lets order only max of 1eth
            
            ticks = {}
            for filt in client.get_symbol_info(current_symbol)['filters']:
                if filt['filterType'] == 'LOT_SIZE':
                    ticks[coin_a] = filt['stepSize'].find('1')
                    if ticks[coin_a] > 0: ticks[coin_a] = ticks[coin_a] - 1
                    print("Lots size:",ticks[coin_a])
                    break
            order_quantity = math.floor(amount * 10**ticks[coin_a]) / float(10**ticks[coin_a])
            if order_quantity > 0:
                print("Trying a buy ",order_quantity,"",coin_a, "at price: ", order_price, "",coin_b)
                try:
                    order = client.order_limit_buy(
                    symbol=current_symbol,
                    quantity=order_quantity,
                    price=order_price)
                except:
                    print("Buy failed")
            else:
                print("Order quantity zero, not trying to buy zero...")
        else:
            print("Not enough balance to buy, balance updated")
       
    if (type == 'sell'):

        global balance_coin_a
        balance = client.get_asset_balance(asset=coin_a)

        amount  = float(balance['free'])
        balance_coin_a = float(balance['free']) 
        if balance_coin_a > min_sell_amount:
            ticks = {}
            priceStep= {}
            for filt in client.get_symbol_info(current_symbol)['filters']:
                if filt['filterType'] == 'LOT_SIZE':
                    ticks[coin_a] = filt['stepSize'].find('1')
                    if ticks[coin_a] > 0: ticks[coin_a] = ticks[coin_a] - 1

            order_quantity = math.floor(amount * 10**ticks[coin_a]) / float(10**ticks[coin_a])

            print("Trying a sell ",order_quantity,"",coin_a, "at price: ", order_price, "",coin_b)
            try:
                order = client.order_limit_sell(
                    symbol=current_symbol,
                    quantity=order_quantity,
                    price=order_price)
                balance = client.get_asset_balance(asset=coin_a)
               
                balance_coin_a = float(balance['free']) 
            except:
                print("Sell failed")
                
        else:
            print("Not enough balance to sell")
       
        
sell_signal=False
buy_signal=False
order_closed = False

init(autoreset=True) # Initialize the CLI and make  functs to clear screen
def move_cursor(x,y):
    print ("\x1b[{};{}H".format(y+1,x+1))
 
def clear():
    print ("\x1b[2J")
 

bot = TradeBot(current_symbol=current_symbol,update_interval=update_interval)
print("Starting tradeBot...")
print("Current symbol:", coin_a, " - ",coin_b)
print("Update interval: ", str(update_interval),"s")
balance_coin_a = client.get_asset_balance(asset=coin_a)
balance_coin_a = float(balance_coin_a['free'])
balance_coin_b = client.get_asset_balance(asset=coin_b)
balance_coin_b = float(balance_coin_b['free'])
starting_price = float(bot.get_latest_price())
starting_capital = balance_coin_a*starting_price + balance_coin_b
starting_capital = 3.05 # REMOVE 

print(coin_a," balance: ", balance_coin_a)
print(coin_b, " balance: ", balance_coin_b)
print("Starting capital:", starting_capital)
print("")
data_update_failed = False
ticks = {}
for filt in client.get_symbol_info(current_symbol)['filters']:
    if filt['filterType'] == 'LOT_SIZE':
        ticks[coin_a] = filt['stepSize'].find('1')-1
        # ticks[coin_a] = filt['stepSize'].find('1') - 2
        print("Lots size:", ticks[coin_a])
        break
min_sell_amount = float(1 / float(10**ticks[coin_a]) )
print("Minimum sell amount: ", min_sell_amount)
def step_size_to_precision(ss):
    return ss.find('1') - 1
def format_value(val, step_size_str):
    precision = step_size_to_precision(step_size_str)
    if precision > 0:
        return "{:0.0{}f}".format(val, precision)
    return math.floor(int(val))
print("Bot Ready.")
time.sleep(1)
clear()
clear_flag = False

while True:
    
    current_time = int(round(time.time()))-time.timezone
   
    try:
        df = bot.get_new_data()
        
        balance_coin_a = client.get_asset_balance(asset=coin_a)
        balance_coin_a = float(balance_coin_a['free'])
        balance_coin_b = client.get_asset_balance(asset=coin_b)
        balance_coin_b = float(balance_coin_b['free'])
        data_update_failed = False
    except:
        print("Data Update failed!")
        if data_update_failed == True:
            print("Data Update failed twice in a row!") 
            print(current_time)
        data_update_failed = True
    data_last_updated=current_time

    # 1. Compute rolling mean
    rm = get_rolling_mean(df["Close"], window=20)
    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df["Close"], window=20)
    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    upper_band = float(upper_band[bot.data_limit-1])
    lower_band= float(lower_band[bot.data_limit-1])

    local_time = time.localtime(current_time)

    current_time_formatted = time.strftime('%m/%d/%Y %H:%M:%S',  time.gmtime(current_time))
    price = float(df["Close"][19])

    total_balance = balance_coin_b + balance_coin_a * price
    total_profit = (total_balance / starting_capital -1) * 100
    price_pos = (price - lower_band) / (upper_band - lower_band)
    print_data = {coin_a:balance_coin_a,coin_b:balance_coin_b}

    # Print data that changes
    if clear_flag == True:
        clear()
        clear_flag = False
    move_cursor(0,0)
    print("Last updated : ",current_time_formatted)
    print(coin_b," \t\t",coin_a)
    print(round(balance_coin_b,6), "\t", round(balance_coin_a,6) )
    print("Current price", price, " (",round(upper_band,7)," / " ,round(lower_band, 7), ") Pos: ",round(price_pos,2))
    print(Fore.GREEN + "Profit: ",Fore.GREEN + str(round(total_profit,2)),Fore.GREEN +"%",Fore.GREEN +"Total : ", Fore.GREEN + str(round(total_balance,2)), Fore.GREEN + coin_b)
    
    if ( price > upper_band):
           
        if (sell_signal==False):
            print(" [Sell signal]",end='')
        sell_signal=True
        buy_signal=False
    elif (price < lower_band):
        if (buy_signal==False):
           print(" [Buy signal]",end='')
        buy_signal = True
        sell_signal=False
    else:
        sell_signal=False
        buy_signal=False
        
    if sell_signal==True and (price / upper_band) < bot.upper_band_m and balance_coin_a > min_sell_amount:
            # Sell balance_coin_a if have
        # print "Selling balance_coin_a"
        if balance_coin_a>min_sell_amount:
            # print sim_time, ':sell at ', price,'usd'
            sell_price = price*1.0
            make_order('sell',sell_price)
            sell_signal=False
    if ( price / lower_band) > bot.lower_band_m and buy_signal == True and balance_coin_b > 0.05:
        # Buy balance_coin_a if funds available
        # print "Buying balance_coin_a"
        if balance_coin_b > 0.01:
            # print sim_time, ':buy at ', price,'usd'
            make_order('buy',price)
            buy_signal=False
        # else:
            # print 'no money'
        
    try:
        open_orders = client.get_open_orders(symbol=current_symbol)
    except:
        print("Trying to get open orders failed")
    if open_orders:
        print("-------------------------------------------------")
        print("Open orders:")
        # print "date  | price | qty | "
        for ord in open_orders:
            print("Order ID: ",ord['orderId'], "Date ",datetime.datetime.fromtimestamp(ord['time']/1000), " | Price: ", ord['price'], " | Qty: ", ord['origQty'])

            order_made_time = ord['time']/1000-time.timezone
            if current_time - order_made_time > 15*60:
                print("Order 15min old - time to cancel")
                client.cancel_order(symbol=current_symbol,orderId=ord['orderId'])
        order_closed = True
        print("-------------------------------------------------------")
    else:
        if order_closed == True:
            print("\n\n\n\n\n")
            balance_coin_a = client.get_asset_balance(asset=coin_a)
            balance_coin_a = float(balance_coin_a['free'])
            balance_coin_b = client.get_asset_balance(asset=coin_b)
            balance_coin_b = float(balance_coin_b['free'])
            order_closed = False
            clear_flag=True # Clear old orders from screen
            
    
    time.sleep(update_interval)    
     
