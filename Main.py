import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.api import OLS
import statsmodels.tools 

#getting FAmafrench data 
pdr.famafrench.get_available_datasets()
start = '1926-01-01'
ff = pdr.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',freq='M',start=start).read()
ff_df = ff[0]
ff_mom_df = pdr.famafrench.FamaFrenchReader('F-F_Momentum_Factor',freq='M',start=start).read()[0]

#plotting cmnds commented out
#ff_df.plot(subplots=True,figsize=(12,4))
#ff_df.rolling(72).mean().plot(subplots=True,figsize=(12,4))
#ff_mom_df.rolling(72).mean().plot(subplots=True,figsize=(12,4))

#merging the both momentum & reader on dates
ffac_merged_df = pd.merge(ff_df,ff_mom_df,on='Date',how = 'inner',sort = True,copy=True,indicator=False,validate='one_to_one')

#getting AAPL from yahoo data
AAPL_df = yf.download('AAPL', start= start)['Adj Close'].resample('M').ffill().pct_change()
AAPL_df = AAPL_df.to_frame()


AAPL_df['str_date'] = AAPL_df.index.astype(str)
AAPL_df['dt_date'] = pd.to_datetime(AAPL_df['str_date']).dt.strftime('%Y-%m')

# Merge risk factor with the live stock data
ffac_merged_df['str_date'] = ffac_merged_df.index.astype(str)
ffac_merged_df['dt_date'] = pd.to_datetime(ffac_merged_df['str_date']).dt.strftime('%Y-%m')
ffy_merged_df = pd.merge(ffac_merged_df,AAPL_df,on='dt_date',how = 'inner',sort = True,copy=True,indicator=False,validate='one_to_one')
# drop extra unnecessary values
ffy_merged_df.drop(columns=['str_date_x','str_date_y'],inplace=True)
#renaming for better stock understanding
ffy_merged_df.rename(columns={'Adj Close' : 'AAPl'}, inplace=True)
ffy_merged_df['AAPl_RF'] = ffy_merged_df['AAPl']*100-ffy_merged_df['RF']
#dropping Nan values , easier to plot with
ffy_merged_df.dropna(axis=0,inplace=True)

# Applying regression
ffy_merged_df.rename(columns={'Mom   ' : 'MOM'},inplace=True)
ffy_merged_df_c = statsmodels.tools.add_constant(ffy_merged_df,prepend = True)
results = OLS(ffy_merged_df_c['AAPL_RF'],ffy_merged_df_c[['const','Mkt-RF','SMB','HML','MOM']],missing = 'drop').fit()
results.summary()