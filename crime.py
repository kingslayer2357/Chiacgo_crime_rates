# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:33:16 2020

@author: kingslayer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet


chicago_df1=pd.read_csv("Chicago_Crimes_2005_to_2007.csv",error_bad_lines=False)
chicago_df2=pd.read_csv("Chicago_Crimes_2008_to_2011.csv",error_bad_lines=False)
chicago_df3=pd.read_csv("Chicago_Crimes_2012_to_2017.csv",error_bad_lines=False)

chicago_df=pd.concat([chicago_df1,chicago_df2,chicago_df3])
chicago_df.shape


#Data Visualisation
chicago_df.isna().any()

chicago_df.drop(columns=["Unnamed: 0","ID","IUCR","FBI Code","Case Number","District","Ward","Community Area","X Coordinate","Y Coordinate","Latitude","Longitude","Location"],inplace=True)
chicago_df.head()

chicago_df.Date=pd.to_datetime(chicago_df.Date,format='%m/%d/%Y %I:%M:%S %p')
chicago_df.index=pd.DatetimeIndex(chicago_df.Date)
chicago_df.Date

chicago_df["Primary Type"].value_counts()
sns.countplot(y="Primary Type",data=chicago_df,order=chicago_df["Primary Type"].value_counts().iloc[:15].index)
chicago_df.columns

sns.countplot(y="Location Description",data=chicago_df,order=chicago_df["Location Description"].value_counts().iloc[:15].index)

chicago_df.resample("Y").size()

plt.plot(chicago_df.resample("Y").size())
plt.title("Crimes by Years")
plt.xlabel("Year")
plt.ylabel("Crimes")

plt.plot(chicago_df.resample("M").size())
plt.title("Crimes by Months")
plt.xlabel("Months")
plt.ylabel("Crimes")

#Preparing data
chicago_prophet=chicago_df.resample("M").size().reset_index()
chicago_prophet.columns=["Date","Crimes"]


chicago_prophet_final=chicago_prophet.rename(columns={"Date":"ds","Crimes":"y"})

m=Prophet()
m.fit(chicago_prophet_final)
future=m.make_future_dataframe(periods=730)
forecast=m.predict(future)
m.plot(forecast,xlabel="Date",ylabel="Crimes")

m.plot_components(forecast)