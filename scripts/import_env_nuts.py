import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates #for working with dates in plots
import seaborn as sns
from calendar import month_name as mn
import datetime
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np

months = mn[1:] #importing month names from calendar

n = 5
#####################################################################################################################
# Daily Syn Concentration
####################################################################################################################


daily_time = pd.read_csv("/D/MIT-WHOI/github_repos/syn_model/data/dailysyntime_matrix.txt",sep = "\t",header = None,names = ["datetime"])
daily_syn = pd.read_csv("/D/MIT-WHOI/github_repos/syn_model/data/dailysynconc_matrix.txt",sep = "\t",header = None,names = ["synconc"])

dfsyn_daily = pd.concat([daily_time,daily_syn],axis = 1)
dfsyn_daily["date"] = pd.to_datetime(dfsyn_daily["datetime"],format='%Y-%m-%d')#reconverting to date column to datetime format
dfsyn_daily["month"] = dfsyn_daily["date"].dt.month #extracting month
dfsyn_daily["doy"] =pd.to_datetime(dfsyn_daily["date"].dt.dayofyear,format = "%j") #extracting day of year
dfsyn_daily["doy_numeric"] =dfsyn_daily["date"].dt.dayofyear #extracting day of year
dfsyn_daily["year"] = dfsyn_daily["date"].dt.year #extracting month
dfsyn_daily["monthname"] = pd.Categorical(dfsyn_daily["date"].dt.month_name(), categories=months, ordered = True)

dfsyn_daily.to_pickle("../data/dfsyn_daily.pkl")

dfsyn_daily.head()

print("printed 1 of", n, "to file in data")

#####################################################################################################################
# High Resolution Environmental Variables
####################################################################################################################



dfenv = pd.read_csv("/D/MIT-WHOI/github_repos/syn_model/data/mvco_env_table.csv")
dfenv["datetime"] = pd.to_datetime(dfenv["time_local"],format='%d-%b-%Y %X')#reconverting to date column to datetime format
dfenv["month"] = pd.to_datetime(dfenv["datetime"].dt.month,format = "%m") #extracting month
dfenv["doy"] = pd.to_datetime(dfenv["datetime"].dt.dayofyear,format = "%j") #extracting day of year
dfenv["doy_numeric"] = dfenv["datetime"].dt.dayofyear #extracting day of year
dfenv["year"] = dfenv["datetime"].dt.year #extracting month
dfenv["monthname"] = pd.Categorical(dfenv["datetime"].dt.month_name(), categories=months, ordered = True)

dfenv.to_pickle("../data/dfenv.pkl")

print("printed 2 of", n, "to file in data")

#####################################################################################################################
# Daily Env. Variables
#####################################################################################################################


dfdaily = pd.read_csv("/D/MIT-WHOI/github_repos/syn_model/data/mvco_daily.csv")
dfdaily["date"] = pd.to_datetime(dfdaily["days"],format='%d-%b-%Y')#reconverting to date column to datetime format
dfdaily["month"] = pd.to_datetime(dfdaily["date"].dt.month,format = "%m") #extracting month
dfdaily["doy"] = pd.to_datetime(dfdaily["date"].dt.dayofyear,format = "%j") #extracting day of year
dfdaily["doy_numeric"] = dfdaily["date"].dt.dayofyear #extracting day of year
dfdaily["year"] = dfdaily["date"].dt.year #extracting month
dfdaily["monthname"] = pd.Categorical(dfdaily["date"].dt.month_name(), categories=months, ordered = True)

dfdaily[["date","Beam_temperature_corrected","AvgSolar","AvgWindSpeed","AvgWindDir","month","doy","doy_numeric","year","monthname"]].to_pickle("../data/mvco_env_daily.pkl")

print("printed 3 of", n, "to file in data")

#####################################################################################################################
# Monthly Nutrients
#####################################################################################################################
heads = ["Event_Num","Event_Num_Niskin","Start_Date","Start_Time_UTC", "Lat","Lon", "Depth", "NO3_a","NO3_b","NO3_c","NH4_a","NH4_b","NH4_c","SiO2_a","SiO2_b","SiO2_c","PO4_a","PO4_b","PO4_c"]

dfnut = pd.read_csv("/D/MIT-WHOI/github_repos/syn_model/data/mvco_nutrients.csv")
dfnut.columns = heads

dfnut["date"] = pd.to_datetime(dfnut["Start_Date"],format='%Y-%m-%d %H:%M:%S.%f').dt.date#reconverting to date column to datetime format
dfnut["time"] = pd.to_datetime(dfnut["Start_Time_UTC"],format='%Y-%m-%d %H:%M:%S.%f').dt.time#reconverting to date column to datetime format
dfnut["datetime"] = dfnut.apply(lambda r : pd.datetime.combine(r['date'],r['time']),1)
dfnut["date"] = dfnut["datetime"].dt.date#reconverting to date column to datetime format
dfnut["month"] = pd.to_datetime(dfnut["datetime"].dt.month,format = "%m") #extracting month
dfnut["doy"] = pd.to_datetime(dfnut["datetime"].dt.dayofyear,format = "%j") #extracting day of year
dfnut["doy_numeric"] = dfnut["datetime"].dt.dayofyear #extracting day of year
dfnut["year"] = dfnut["datetime"].dt.year #extracting month
dfnut["NO3_mean"] = dfnut[["NO3_a","NO3_b","NO3_c"]].mean(axis = 1)
dfnut["NH4_mean"] = dfnut[["NH4_a","NH4_b","NH4_c"]].mean(axis = 1)
dfnut["PO4_mean"] = dfnut[["PO4_a","PO4_b","PO4_c"]].mean(axis = 1)
dfnut["SiO2_mean"] = dfnut[["SiO2_a","SiO2_b","SiO2_c"]].mean(axis = 1)

dfnut.to_pickle("../data/dfnut.pkl")

print("printed 4 of", n, "to file in data")

##################################################################################################################### merging daily syn, daily environmental variables and monthly nutrient concentrations
#####################################################################################################################
dfdaily.set_index("date", inplace = True)
dfsyn_daily.set_index("date", inplace = True)
dfnut.set_index("date",inplace = True)

dfdaily = dfdaily.dropna()
dfsyn_daily = dfsyn_daily.dropna()

dfmerged = dfdaily[["Beam_temperature_corrected","AvgSolar","AvgWindSpeed","AvgWindDir","month","doy","doy_numeric","year","monthname"]].combine_first(dfsyn_daily[["synconc"]]).combine_first(dfnut[["NO3_mean","NH4_mean","PO4_mean","SiO2_mean"]])
dfmerged.to_csv("../data/dfmerged_dailysynenv.csv")

print("printed 5 of", n, "to file in data")
