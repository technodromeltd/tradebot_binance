# tradebot_binance
Trading bot using Binance API to make real trades based on Bollinger bands analysis.

***USE AT YOUR OWN RISK***

Trading bot using Binance API to make real trades based on Bollinger bands analysis.
Can only trade between two assets using limit bids, where the sell price need to be higher than current price,  
so in theory cant lose all of your money by mistakenly selling too low :)

You need to set up binance api account first and add your key and secret to binance_api_keys.py file e.g.
    
    #/binance_api_keys.py
    api_key = 'YOURKEY'
    secret_key = 'YOURSECRET'

How to create binance apikey https://www.binance.com/en/support/articles/360002502072

Dependeciens:
- python-binance https://python-binance.readthedocs.io/en/latest/overview.html
- pandas

Usage:
>main.py  < COINPAIR > < UPDATE_INTERVAL_S >

Arguments:
-  < COINPAIR >          : e.g. "LTCETH" (default)
-  < UPDATE_INTERVAL_S > : e.g. 15 (default)


Other Settings:
   -  upper_band_m        : Selling when price is higher than upper_band_m * bollinger bands upper value (current_price > upper_band_m * upper_band) *This value is based on backprogatation tests on past data  
   -  lower_band_m        : Buying when (current_price < lower_band_m * lower_band) *This value is based on backprogatation tests on past data  
   -  data_limit          : How many past kline price data intervals to get
    - data_kline_interval : What kline price data interval to use, KLINE_INTERVAL_1MINUTE to 1 KLINE_INTERVAL_1MONTH 
        


