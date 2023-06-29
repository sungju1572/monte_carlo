import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import talib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier
import datetime
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import matplotlib.pyplot as plt




a = pd.read_csv("gan_result_fid_0015_epoch_200.csv")
b = pd.read_csv("실제데이터_kospi30_6_15.csv")


a.columns
a = a.drop(["Unnamed: 0"], axis=1)
b = b.drop(["Unnamed: 0"], axis=1)



a["trade_count"] = a["trade_count"].astype(int)


a[a["count"]==200].groupby(a["model"]).mean().iloc[0]["payoff_ratio"]
a[a["count"]==100].groupby(a["model"]).mean().iloc[0]["payoff_ratio"]
a[a["count"]==50].groupby(a["model"]).mean().iloc[0]["payoff_ratio"]
a[a["count"]==20].groupby(a["model"]).mean().iloc[0]["payoff_ratio"]



x = ["real_data", "20_count","50_count","100_count","200_count"]

#payoff
dt_list = [a[a["count"]==200].groupby(a["model"]).mean().iloc[0]["payoff_ratio"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[0]["payoff_ratio"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[0]["payoff_ratio"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[0]["payoff_ratio"], 1.183]

lg_list = [a[a["count"]==200].groupby(a["model"]).mean().iloc[1]["payoff_ratio"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[1]["payoff_ratio"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[1]["payoff_ratio"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[1]["payoff_ratio"],1.232]

rf_list = [a[a["count"]==200].groupby(a["model"]).mean().iloc[2]["payoff_ratio"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[2]["payoff_ratio"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[2]["payoff_ratio"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[2]["payoff_ratio"],1.060]


xg_list = [a[a["count"]==200].groupby(a["model"]).mean().iloc[3]["payoff_ratio"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[3]["payoff_ratio"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[3]["payoff_ratio"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[3]["payoff_ratio"],1.126]

#pf
dt_list2 = [a[a["count"]==200].groupby(a["model"]).mean().iloc[0]["profit_factor"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[0]["profit_factor"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[0]["profit_factor"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[0]["profit_factor"],1.175]

lg_list2 = [a[a["count"]==200].groupby(a["model"]).mean().iloc[1]["profit_factor"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[1]["profit_factor"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[1]["profit_factor"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[1]["profit_factor"],1.628]

rf_list2 = [a[a["count"]==200].groupby(a["model"]).mean().iloc[2]["profit_factor"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[2]["profit_factor"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[2]["profit_factor"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[2]["profit_factor"],1.094]


xg_list2 = [a[a["count"]==200].groupby(a["model"]).mean().iloc[3]["profit_factor"],
a[a["count"]==100].groupby(a["model"]).mean().iloc[3]["profit_factor"],
a[a["count"]==50].groupby(a["model"]).mean().iloc[3]["profit_factor"],
a[a["count"]==20].groupby(a["model"]).mean().iloc[3]["profit_factor"],1.1620]






##실제 데이터 시각화
b = pd.read_csv("실제데이터_kospi30_6_15.csv")

b = b.drop(["Unnamed: 0"], axis=1)


bb = b.groupby(b["id"]).mean()[["trade_count","winning_ratio", "payoff_ratio","profit_factor"]]


##생성 데이터 비교
a = pd.read_csv("gan_result_fid_0015_epoch_200_2.csv")
a = a.drop(["Unnamed: 0"], axis=1)

#1. 각 종목별 전체 평균 비교 (vs. 실제 데이터)

a[a["count"]==200].groupby(a["model"]).mean().iloc[0]["profit_factor"]
a[a["count"]==150].groupby(a["model"]).mean().iloc[0]["profit_factor"]
a[a["count"]==100].groupby(a["model"]).mean().iloc[0]["profit_factor"]
a[a["count"]==50].groupby(a["model"]).mean().iloc[0]["profit_factor"]
a[a["count"]==20].groupby(a["model"]).mean().iloc[0]["profit_factor"]



a_group = a.groupby(['ticker','count']).mean()

aa = a.groupby(['ticker']).mean()

x = ["trade_count", "winning_ratio", "payoff_ratio", "profit_factor"]

aa[aa.index == 270]

#plot
import matplotlib.pyplot as plt
import numpy as np



aa = aa.rename(columns={"trade_count" : "trade_count_create" , "winning_ratio" : "winning_ratio_create", 
                        "payoff_ratio" : "payoff_ratio_create" , "profit_factor" : "profit_factor_create"})  



concat = pd.concat([bb,aa[["trade_count_create", "winning_ratio_create", "payoff_ratio_create", "profit_factor_create"]]], axis=1)

concat = concat.reset_index()

concat['index'] = concat['index'].astype(str)

#거래횟수
##1
plt.bar(concat["index"], concat["trade_count"],  alpha=0.4, color='red', label='Real')
plt.bar(concat["index"], concat["trade_count_create"], alpha=0.4, color='blue', label='Generate')

plt.title("trade_count")
plt.xticks(rotation=45)
plt.legend()
plt.show()

##2
plt.bar(concat.index-0.175, concat["trade_count"], width = 0.4, alpha=0.4, color='red', label='Real')
plt.bar(concat.index+0.175, concat["trade_count_create"], width = 0.4,  alpha=0.4, color='blue', label='Generate')

plt.title("trade_count")
plt.xticks(ticks = concat.index ,labels = concat["index"],rotation=45)
plt.legend()
plt.show()

##4
plt.boxplot([concat["trade_count"],concat["trade_count_create"]])

plt.title("trade_count")
plt.xticks([1,2], ["Real", "Generate"])
plt.show()

#winning_ratio
##1
plt.bar(concat["index"], concat["winning_ratio"],  alpha=0.4, color='red', label='Real')
plt.bar(concat.index, concat["winning_ratio_create"], alpha=0.4, color='blue', label='Generate')

plt.title("winning_ratio")
plt.xticks(rotation=45)
plt.legend()
plt.show()

##2
plt.bar(concat.index-0.175, concat["winning_ratio"], width = 0.4, alpha=0.4, color='red', label='Real')
plt.bar(concat.index+0.175, concat["winning_ratio_create"], width = 0.4,  alpha=0.4, color='blue', label='Generate')

plt.title("winning_ratio")
plt.xticks(ticks = concat.index ,labels = concat["index"],rotation=45)
plt.legend()
plt.show()

##3
plt.scatter(concat.index, concat["winning_ratio"], alpha=0.4, color='red', label='Real')
plt.scatter(concat.index, concat["winning_ratio_create"], alpha=0.4, color='blue', label='Generate')


plt.title("winning_ratio")
plt.xticks(ticks = concat.index ,labels = concat["index"],rotation=45)
plt.legend()
plt.show()

##4
plt.boxplot([concat["winning_ratio"],concat["winning_ratio_create"]])

plt.title("winning_ratio")
plt.xticks([1,2], ["Real", "Generate"])
plt.show()


##payoff_ratio
#1
plt.bar(concat["index"], concat["payoff_ratio"],  alpha=0.4, color='red', label='Real')
plt.bar(concat.index, concat["payoff_ratio_create"], alpha=0.4, color='blue', label='Generate')

plt.title("payoff_ratio")
plt.xticks(rotation=45)
plt.legend()
plt.show()

##2
plt.bar(concat.index-0.175, concat["payoff_ratio"], width = 0.4, alpha=0.4, color='red', label='Real')
plt.bar(concat.index+0.175, concat["payoff_ratio_create"], width = 0.4,  alpha=0.4, color='blue', label='Generate')

plt.title("payoff_ratio")
plt.xticks(ticks = concat.index ,labels = concat["index"],rotation=45)
plt.legend()
plt.show()

##3
plt.scatter(concat.index, concat["payoff_ratio"], alpha=0.4, color='red', label='Real')
plt.scatter(concat.index, concat["payoff_ratio_create"], alpha=0.4, color='blue', label='Generate')

#plt.plot(concat.index, concat["payoff_ratio"])

plt.title("payoff_ratio")
plt.xticks(ticks = concat.index ,labels = concat["index"],rotation=45)
plt.legend()
plt.show()

##4
plt.boxplot([concat["payoff_ratio"],concat["payoff_ratio_create"]])

plt.title("payoff_ratio")
plt.xticks([1,2], ["Real", "Generate"])
plt.show()


##profit_factor
#1
plt.bar(concat["index"], concat["profit_factor"],  alpha=0.4, color='red', label='Real')
plt.bar(concat.index, concat["profit_factor_create"], alpha=0.4, color='blue', label='Generate')

plt.title("profit_factor")
plt.xticks(rotation=45)
plt.legend()
plt.show()

##2
plt.bar(concat.index-0.175, concat["profit_factor"], width = 0.4, alpha=0.4, color='red', label='Real')
plt.bar(concat.index+0.175, concat["profit_factor_create"], width = 0.4,  alpha=0.4, color='blue', label='Generate')

plt.title("profit_factor")
plt.xticks(ticks = concat.index ,labels = concat["index"],rotation=45)
plt.legend()
plt.show()

##3
plt.scatter(concat.index, concat["profit_factor"], alpha=0.4, color='red', label='Real')
plt.scatter(concat.index, concat["profit_factor_create"], alpha=0.4, color='blue', label='Generate')

#plt.plot(concat.index, concat["payoff_ratio"])

plt.title("profit_factor")
plt.xticks(ticks = concat.index ,labels = concat["index"],rotation=45)
plt.legend()
plt.show()

##4
plt.boxplot([concat["profit_factor"],concat["profit_factor_create"]])

plt.title("profit_factor")
plt.xticks([1,2], ["Real", "Generate"])
plt.show()

