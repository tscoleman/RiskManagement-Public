##textbook Ch10 tables 
import RiskMgmtFunctions  as fns
import SamplePort_define as sec_master

import numpy as np
import math
from scipy import stats
from scipy.linalg import solve
import scipy.stats as stats
import locale
import math
from scipy.stats import norm, t
import ipywidgets as widgets
from babel import numbers
from tabulate import tabulate
import pandas as pd
import io
###Reports for QRM Sample Portfolio
##Set holdings
seclist = ["10yr UST", "5yr Bond Opt", "10yr UST", "10yr Swap", "10yr UKG", "FTE CDS", "CAC Index Futures", "FTE Equity", "GBPCash", "EURCash", "5yr UST"]
holdings = [20, -20, 30, -20, 25, 55, 7, 5, -10, 0, 0]
subportfolio = ["Government", "Government", "Swaps", "Swaps", "Government", "Credit", "Equity", "Equity", "Government", "Equity", "Government"]

#############################Introduction##########################
n = 9
maximum = max(holdings)
minimum = min(holdings)
xx = ["$", "¥", "£", "€"]
yy = ['USD', 'JPY', 'GBP', 'EUR']
df_sec = sec_master.df_sec
df_market = sec_master.df_market
df_rf=sec_master.df_rf
currlist = seclist[:]

for i in range(len(seclist)):
    x2 = fns.fxrateSec(seclist[i], df_sec, df_market)
    currency_map = dict(zip(yy, xx))
    x2[0] = currency_map.get(x2[0], x2[0])
    currlist[i] = [item for item in xx if item == x2[0]]
    currlistvalue = currlist[:]
    currlistvalue[i] = str(currlist[i][0])


table = [widgets.Label(value="Security"), widgets.Label(value="Amount (mn local)"), widgets.Label(value="Curr"), widgets.Label(value="Subportfolio")]
for i in range(n):
    row = [
        widgets.Dropdown(options=seclist, value=seclist[i]),
        widgets.FloatSlider(value=holdings[i], min=minimum, max=maximum, step=0.1, readout_format='.1f', description=""),
        widgets.Dropdown(options=currlistvalue, value=currlistvalue[i]),
        widgets.Dropdown(options=subportfolio, value=subportfolio[i])
    ]
    table.extend(row)

#grid = widgets.GridBox(table, layout=widgets.Layout(grid_template_columns="repeat(4, 300px)"))
from IPython.display import display
#display(grid)


###Table 1 - Securities, Risk Factors, and DV01s
settledate = fns.datetime(2009, 1, 27, 0, 0, 0)
df_sec=sec_master.df_sec
df_market=sec_master.df_market
df_rf=sec_master.df_rf
displayRF=fns.displayRF(seclist, holdings, settledate)
displayRF
# Save the tabulated output to an Excel file
#with open('output.txt', 'w') as f:
#    f.write(displayRF)
#df = pd.read_csv('output.txt', sep="|")
#df.to_excel('output.xlsx', index=False)


##Create report tables
settledate = fns.datetime(2009, 1, 27, 0, 0, 0)
##nav=1 for dollar term, nav=500000 for percentage term 
formatted_xx = fns.reportFunction1(seclist, holdings, settledate,nav=1)
yy = fns.volbySubPortfolio(seclist, holdings, settledate, subportfolio)


t1 = tabulate(formatted_xx[0], headers='firstrow', tablefmt='grid')
print(t1)


t1 = tabulate(formatted_xx[1], headers='firstrow', tablefmt='grid')
print(t1)

t1 = tabulate(formatted_xx[2], headers='firstrow', tablefmt='grid')
print(t1)


t1 = tabulate(formatted_xx[3], headers='firstrow', tablefmt='grid')
print(t1)


t1 = tabulate(formatted_xx[4], headers='firstrow', tablefmt='grid')
print(t1)

t1 = tabulate(yy, headers='firstrow', tablefmt='grid')
print(t1)


result = fns.varsumm(seclist, holdings, settledate)
formatted_result = []
for item in result:
    if isinstance(item[1], (int, float)):
        formatted_value = f"{item[1]:,}"
        formatted_result.append([item[0], formatted_value])
    else:
        formatted_result.append(item)
table = []
for row in formatted_result:
    table.append(row)
formatted_table = tabulate(table, headers='firstrow', tablefmt='grid')
print(formatted_table)
