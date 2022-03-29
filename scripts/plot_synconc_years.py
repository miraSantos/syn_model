import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates #for working with dates in plots
import seaborn as sns
from calendar import month_name as mn
from matplotlib.dates import MonthLocator, DateFormatter

months = mn[1:]
print(months)

date = pd.read_csv("/D/MIT-WHOI/github_repos/syn_model/data/alldatenum.txt",sep = "/t",header = None,names = ["date"])
synconc = pd.read_csv("/D/MIT-WHOI/github_repos/syn_model/data/allsynconc.txt",sep = "/t",header = None,names = ["synconc"])
df = pd.concat([date,synconc],axis = 1)
df["date"] = pd.to_datetime(df["date"],format='%Y-%m-%dT%X')#reconverting to date column to datetime format
df["month"] = df["date"].dt.month #extracting month
df["doy"] =df["date"].dt.dayofyear #extracting day of year
df["year"] = df["date"].dt.year #extracting month
df["monthname"] = pd.Categorical(df["date"].dt.month_name(), categories=months, ordered = True)#extracting month name

print(df.shape)
df.head()

ii = 1
for year in df["year"].unique():
    plt.figure(figsize=(15, 8), dpi=1000)

    # set font size and grid style
    sns.set(font_scale = 2, style = "whitegrid")

    #set font
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]})

    sns.relplot(kind='line', data=df[df["year"] == year], x='doy', y='synconc', hue='year',
                    aspect=2.5,
                    marker='.',
                palette = "deep")
    plt.gca().xaxis.set_major_locator(MonthLocator())  # Tick locator
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b'))
    plt.xlabel("Day of the Year")
    plt.ylabel("Concentration $(cells$ $mL^{-1})$")
    plt.title( str(year) + " $\it{Syn}$ Concentration")
    plt.gca().set_xlim([1,366])
    plt.gca().set_ylim([0,400000])
    plt.savefig("/D/MIT-WHOI/github_repos/syn_model/results/syn_conc/"+str(year) + "_synconc"+'.jpg')
    print("saved " + str(ii) + " of " + str(df["year"].nunique()))
    ii = ii +1
    plt.close("all")

