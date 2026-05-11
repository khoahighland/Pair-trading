import pandas as pd
import yfinance as yf
import time

start = time.perf_counter()

start = '2023-01-01'
end = '2025-12-31'

tickers = pd.read_csv('tickers.csv')
tickers = tickers.columns.tolist()

data = yf.download(tickers,
                   start= start,
                   end= end,
                   )['Close']

# data.to_csv('SP500.csv')
print(data)

end = time.perf_counter()
elapsed_time = end - start
print(f"Total running time: {elapsed_time:.4f} seconds")

