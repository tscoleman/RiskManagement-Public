 #Dataframe function
####Various Functions for Risk Management Manuscript

import SamplePort_df_python_define as sec_master

import numpy as np
from functools import reduce
import math
import scipy
from scipy import optimize
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
###Dates
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def serialday(date):
    """Input a date object, returns number of days since 30-dec-1899"""
    xdate = date
    if isinstance(date, list):    ## if a list, then need to convert to a proper date
        xdate = datetime(*date)
    delta = xdate - datetime(1899, 12, 30)
    return delta.days ##  should not add two. That is inherited from Mathematica where it was the number of seconds since some date, and for some reason reuired the "+ 2" to work properly


def dateday(serialday):
    return datetime(1900, 1, 1) + timedelta(days=serialday - 2)

###PV, DV01, and Portfolio Functions
##Valuation functions

#Bond and rate PV functions
def pv(rate, nper, pmt, fv):
    return pmt*(1 - (1/((1 + rate)**nper)))/rate + fv/((1 + rate)**nper)

def bondprcSimp(yld, settledate, parms):
    def serialday(date):
        delta = date - datetime(1899, 12, 30)
        return delta.days + 2
        pass
    
    yldperiod = yld / parms[2]
    nper = round((serialday(datetime(*parms[1])) - serialday(settledate)) / (365 / parms[2]))
    pmt = parms[0] / parms[2]
    return pv(yldperiod, nper, pmt, 100)

def bondprc(yld, settledate, parms):
    yldperiod = yld / parms[2]
    if parms[2] == 1:
        hlfyr = 12
    else:
        hlfyr = 6
    matsd = serialday(datetime(*parms[1]))
    d2 = list(parms[1])     ### use list(parm[1]) instead of parm[1], or the original parm[1] will be also changed when we change d2 later
    #print(d2)
    #print(parms[1])
    date_list = [
        settledate.year,  # Year
        settledate.month,  # Month
        settledate.day,  # Day
        settledate.hour,  # Hour
        settledate.minute,  # Minute
        settledate.second,  # Second
    ]
    
    d2[0] = date_list[0]   ### update the YEAR of maturity date with the YEAR of settlement date
    d2 =datetime(*d2)
    #print(date_list[0])
    d0 = d2 - relativedelta(months=+12)   ### 12months before the UPDATED maturity date
    d1 = d2 - relativedelta(months=+hlfyr) ### a payment peroid before the UPDATED maturity date
    d3 = d2 + relativedelta(months=+hlfyr) ### a payment peroid after the UPDATED maturity date
    d4 = d2 + relativedelta(months=+12)   ### 12months after the UPDATED maturity date
    
    d0sd = serialday(d0)
    d1sd = serialday(d1)
    d2sd = serialday(d2)
    d3sd = serialday(d3)
    d4sd = serialday(d4)
    
    settlesd = serialday(settledate)
    #print()
    if d0sd < settlesd <= d1sd:             #### Now we have set fixed d0-d4,and split 24months into 4 ranges, and we have to know which range the settlement day will be. 
        dbefore = d0sd                      ### And DBEFORE will be set as the start value of that range.
        dafter = d1sd                       ### And DAFTER will be set as the end value of that range.
    elif d1sd < settlesd <= d2sd:
        dbefore = d1sd
        dafter = d2sd
    elif d2sd < settlesd <= d3sd:
        dbefore = d2sd
        dafter = d3sd
    elif d3sd < settlesd <= d4sd:
        dbefore = d3sd
        dafter = d4sd
    
    dayfrac = (dafter - settlesd) / (dafter - dbefore)
    nper = round((matsd - dafter) / (365/parms[2]))
    #print(matsd)
    pmt = parms[0] / parms[2]
    bondpv = pv(yldperiod, nper, pmt, 100)
    bondpv = (pmt + bondpv) / ((1 + yldperiod) ** dayfrac)    ### discount the "PV dafter + next PMT after Settlement day" to the settlement day.

    if len(parms) == 5:
        floatreset = parms[3] / parms[2]
        bondpv = bondpv - (floatreset + 100) / ((1 + yldperiod) ** dayfrac)
    
    ai = pmt * (1 - dayfrac)
    prc = bondpv - ai
    
    return [prc, ai, bondpv]


def bondpv(inputs, settledate, holding, parms):
    """Inputs are [yld,FX rate]"""
    bondprc_result = bondprc(inputs[0], settledate, parms)
    return bondprc_result[2] * holding * 10000 / inputs[1]


##swappvactual is calculating price of swap,like TVM menu, no specific date
def swappvactual(inputs, settledate, holding, parms):
    """swappvactual is calculating price of swap,like TVM menu, no specific date. 
    input=[yeild,spread,FX rate]"""
    bond_price = bondprc(inputs[0] + inputs[1], settledate, parms)[2]
    return bond_price * holding * 10000 / inputs[2]

####swappv is calculating value of swap,like Bond menu,add date
def swappv(inputs, settledate, holding, parms):
    """Swappv is calculating value of swap,like Bond menu,add date. 
    input=[yeild,spread,FX rate]"""
    yld = inputs[0] + inputs[1]
    yldperiod = yld / parms[2]
    nper = parms[1] * parms[2]
    pmt = parms[0] / parms[2]
    xx = (pv(yldperiod, nper, pmt, 100) - 100) * holding * 10000 / (inputs[2])
    #10000=holding/notion=1million/100
    return xx


#Bond option function
from scipy.stats import norm

def bondoptionprc(expiry, rshort, fyld, strike, vol, underlierTerm, underlierFreq, putcall):
    """bondoptionprc() calculates the PV for a bond option (BS option on rate).  
    input=[Expiry,rshort,Forward yield,Strike yield,Volatility,Underlier Term,Underlier Frequency,"P"/"C"]
    Inputs are:
 1. Expiry (years)  
 2. rshort (rate, decimal like 0.02 for 2%, quoted at bond frequency)
 3. Forward yield (rate, decimal like .05 for 5%, bond freq)
 4. Strike yield (rate, decimal like 0.04 for 4%, bond freq)
 5. Volatility (decimal like 0.20 for 20%)
 6. Underlier Term
 7. Underlier Frequency
 8. P=put = payers option (put on the bond, call on rates, valuable when rates are high or prices low), "C"=call = receiver's option (call on bond price)
    expiry = inputs[0]
    rshort = inputs[1]
    fyld = inputs[2]
    strike = inputs[3]
    vol = inputs[4]
    underlierTerm = inputs[5]
    underlierFreq = inputs[6]
    putcall = inputs[7]"""
    #expiry=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][0]
    #rshort=sec_master.df_sec.loc['5yr Bond Opt']['InputsValue'][0]
    #fyld=sec_master.df_sec.loc['5yr Bond Opt']['InputsValue'][1]
    #strike=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][1]
    #vol=sec_master.df_sec.loc['5yr Bond Opt']['InputsValue'][2]
    #underlierTerm=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][2]
    #underlierFreq=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][3]
    #putcall=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][4]
    pvann = pv(fyld/underlierFreq, underlierFreq*underlierTerm, 1/underlierFreq, 0)
    pvann = pvann / ((1 + rshort/underlierFreq) ** (underlierFreq*expiry))
    
    d1 = (np.log(fyld/strike) + (expiry*(vol**2))/2) / (vol*np.sqrt(expiry))
    d2 = d1 - vol*np.sqrt(expiry)
    p1 = norm.cdf(d1)
    p2 = norm.cdf(d2)
    
    if putcall == 'P':
        return pvann*(fyld*p1 - strike*p2)*100
    else:
        return pvann*(fyld*(p1 - 1) - strike*(p2 - 1))*100

def bondoptionpv(inputs, settledate, holding, parms):
    """Inputs=[rshort,Forward yield,Volatility,FX rate]  """
    return bondoptionprc(parms[0], inputs[0], inputs[1], parms[1], inputs[2],parms[2], parms[3], parms[4])* holding*10000 / inputs[3]

#CDS PV functions
def simpCDSpv(inputs, settledate, holding, parms):
    """The inputs are:  1. risk-free rate  2. market spread   3. FX rate
        The parms are:
    	1. CDS coupon (spread)
    	2. Number of years
    	3. Coupon frequency (so total no. periods = parms[[2]]*parms[[3]])
        4. Currency"""
    yldmkt = inputs[0] + inputs[1]
    yldperiod = yldmkt / parms[2]
    nper = parms[1] * parms[2]
    pmt = (parms[0] + 100 * inputs[0]) / parms[2]
    xx = (pv(yldperiod, nper, pmt, 100) - 100) * holding * 10000 / inputs[2]
    return xx



#Cash PV function
def cashpv(inputs, settledate, holding, parms):
    """The input is FX rate"""
    return holding * 1000000 / inputs[0]

#Equity PV functions
#inputs contains the equity index and the FX rate.
def eqtypv(inputs, settledate, holding, parms):
    """Inputs are [Index Level, FX rate]"""
    x = len(parms)
    if x < len(inputs):
        return "Error in eqtypv"
    else:
        return holding * 1000000 * reduce(lambda a, b: a * b, map(lambda a, b: a ** b, inputs[:x-1], parms[:2-x])) / inputs[-1]

def eqtyFutpv(inputs, settledate, holding, parms):
    x = len(parms)
    marketDataNames=sec_master.df_market['marketDataNames'].tolist()  
    if x < len(inputs):
        return "Error in eqtyFutpv"
    else:
        return (holding * 1000000 * 
                reduce(lambda a, b: a * b, 
                       map(lambda a, b: a ** b, 
                           inputs[:x-1], parms[:x-1]))/ (sec_master.df_market['marketDataValues'][marketDataNames.index(parms[1] + "FX")]))

#def eqtypvOld(inputs, settledate, holding, parms):
#    return inputs[0] * holding * parms[0] * 1000000 / inputs[-1]


#def eqtyFutpvOld(inputs, settledate, holding, parms):
#    return inputs[0] * holding * 1000000 * parms[0] / next(x for x, y in zip(sec_master.marketDataValues, sec_master.marketDataNames) if y == parms[-1] + "FX")

#Futures PV functions
def futpv(inputs, settledate, holding, parms):
    """Inputs = [Future Price],Parms = [Contract size, Currency]"""
    return inputs[0] * holding * 1000000 * parms[0] / next(
        marketDataValue for marketDataName, marketDataValue in zip(sec_master.df_market['marketDataNames'], sec_master.df_market['marketDataValues'])
        if parms[-1] in marketDataName and "FX" in marketDataName
    )

##utility function
def fxrateSec(sec, df_sec, df_market):    
    xnames = sec_master.df_sec.loc[sec]['secParmNames']
    xvals = sec_master.df_sec.loc[sec]['secParmValues']    
    xx1 = xvals[xnames.index('Currency')]
    xfx = sec_master.df_market.loc[df_market['marketDataNames'] == (xx1 + "FX"), 'marketDataValues'].values[0]
    return [xx1,xfx]


#dv01
def dv01posn(security, settledate, riskfactor, holding):
    """Single security(only eligable for single riskfactor) DV01 functions,a function to calculate dv01 for a position(single security& single riskfactor)"""
    count = 1
    # check to make sure risk factors, etc., have been set up in global lists
    count = sec_master.df_market['marketDataNames'].value_counts()[riskfactor]
    count = count*sec_master.df_rf['riskFactorNames'].value_counts()[riskfactor]
    count = count*sec_master.df_sec['secNames'].value_counts()[security]
    if count == 0:
        return "Error"
    else:
        # First get the input and bump up and down
        parms = sec_master.df_sec.loc[security]['secParmValues']
        in_val = sec_master.df_market.loc[riskfactor]['marketDataValues']
        h = sec_master.df_rf.loc[riskfactor]['riskFactorBumpSize']
        leveltst = sec_master.df_rf.loc[riskfactor]['riskFactorBumpTypes']

        if leveltst == "level":
            inup = in_val + h
            indn = in_val - h
            h = 2*h
        else:
            inup = in_val*(1 + h)
            indn = in_val/(1 + h)
            h = (1 + h) - (1/(1 + h))
        
        # Now decide where to put it in the input list
        mktinplist = sec_master.df_sec.loc[security]['secMktInputNames']
        count = len(mktinplist)
        count = count* mktinplist.count(riskfactor)
        
        if count == 0:
            return "Error in dv01posn"
        else:
            #xpvfn = getattr(DistFn_df_python_function, sec_master.df_sec.loc[security]['secPVFunction'])
            xpvfn = globals()[sec_master.df_sec.loc[security]['secPVFunction']]
            mktinpval = [sec_master.df_market['marketDataValues'][i] for name in mktinplist for i in [j for j, n in enumerate(sec_master.df_market['marketDataNames']) if n == name]]
            i = mktinplist.index(riskfactor)
            mktinpval[i] = inup
            xup = xpvfn(mktinpval, settledate, holding, parms)
            mktinpval[i] = indn
            xdn = xpvfn(mktinpval, settledate, holding, parms)
            y = sec_master.df_rf.loc[riskfactor]['riskFactorBumpUnit']
            sign_adjt = np.sign(holding)
            x = y*(xup - xdn)/h*sign_adjt
            
            return x
        
def pvposn(security, holding, settledate, marketDataValues):
    """Function to calculate the value of a position."""
    parms = sec_master.df_sec.loc[security]['secParmValues']
    mktinplist = sec_master.df_sec.loc[security]['secMktInputNames']
    mktinpval=[marketDataValues.loc[name]['marketDataValues']for name in mktinplist]
    xpvfn = globals()[sec_master.df_sec.loc[security]['secPVFunction']]
    pv = xpvfn(mktinpval, settledate, holding, parms)
    return pv        
        
def pvPort(seclist, holdings, settledate, marketDataValues):
    """Function to loop over securities and holdings in a portfolio.
    Need to have lists (securities and holdings) first, then non-lists (settledate, etc) """          
    def pvposn2(security, holding):
        return pvposn(security, holding, settledate,marketDataValues)
    return list(map(pvposn2, seclist, holdings))  

def dv01sec(security, settledate, riskfactor):
    """Function to calulate the price change for a single security in terms of a certain risk factor change"""
    y = sec_master.df_rf.loc[riskfactor]['riskFactorMult']
    return y * dv01posn(security, settledate, riskfactor,1) 

#Security list DV01 functions
def populateSecDV01s(seclist, holdings, settledate, df_sec, df_rf, df_market):
    #seclist = df_sec[df_sec['secNames'].isin(seclist)]['secNames'].tolist()    
    secDV01Names = [df_sec.loc[i]['secRiskFactorNames']for i in seclist] # This creates a list of security rfs - one list for each security. This way of doing it seems clumsy, but it essentially "loops through" seclist.
    #we need create three new list of list(secDV01Values,xup,xdn) to avoid the overwriting. 
    secDV01Values = [sublist.copy() for sublist in secDV01Names]
    xup = [sublist.copy() for sublist in secDV01Names]
    xdn = [sublist.copy() for sublist in secDV01Names]
    flat_list=[item for sublist in secDV01Names for item in sublist]
    def remove_duplicates(lst): 
        seen = set() 
        result = [] 
        for item in lst: 
            if item not in seen: 
                result.append(item) 
                seen.add(item) 
        return result
    rflist = list(remove_duplicates(flat_list))    
    countrf = len(rflist)
    countsec = len(seclist)
    if countrf * countsec == 0: 
        return "Error in populateSecDV01s - seclist or rflist empty"
    # If OK then do the work of the function.
    for i in range(countrf):  # Loop over risk factors.
        xmarketDataup = df_market.copy()
        xmarketDatadn = df_market.copy()
        # Populate the local versions of market data - later the bumped RF values will be substituted.
        irfmkt=df_market.loc[rflist[i]]['marketDataNames'] # First get the input and bump up and down.
        in_val = df_market.loc[irfmkt]['marketDataValues']
        irfmkt=df_rf.loc[rflist[i]]['riskFactorNames'] 
        h=df_rf.loc[rflist[i]]['riskFactorBumpSize'] 
        leveltst=df_rf.loc[rflist[i]]['riskFactorBumpTypes']
        bumpunit = df_rf.loc[rflist[i]]['riskFactorBumpUnit']
        if leveltst == "level":
            inup = in_val + h
            indn = in_val - h
            h = 2 * h  # Put up & dn into local market data.
            xmarketDataup.loc[irfmkt,'marketDataValues'] = inup
            xmarketDatadn.loc[irfmkt,'marketDataValues']= indn
        elif leveltst == "percent":
            inup = in_val * (1 + h)
            indn = in_val / (1 + h)
            h = (1 + h) - (1 / (1 + h))
            xmarketDataup.loc[irfmkt,'marketDataValues']= inup
            xmarketDatadn.loc[irfmkt,'marketDataValues']= indn
        for j in range(countsec): 
            isec = sec_master.df_sec['secNames'].tolist().index(seclist[j])   
            if rflist[i] in df_sec['secRiskFactorNames'][isec]:
                irfsec = df_sec['secRiskFactorNames'][isec].index(rflist[i])           
                xup[j][irfsec] = pvposn(seclist[j], holdings[j], settledate, xmarketDataup)
                xdn[j][irfsec] = pvposn(seclist[j], holdings[j], settledate, xmarketDatadn)               
                secDV01Values[j][irfsec] = bumpunit * (xup[j][irfsec] - xdn[j][irfsec]) / h
            else:
                pass
    secDV01s = [seclist, secDV01Names, secDV01Values]
    return secDV01s


def populatePortDV01s(seclist, holdings, settledate, df_sec, df_rf, df_market):
    secDV01s = populateSecDV01s(seclist, holdings, settledate,df_sec, df_rf, df_market)
    riskFactorNames=df_rf['riskFactorNames'].tolist()
    portDV01Names = riskFactorNames.copy() # Later the empty elements will be discarded
    portDV01Values = [0]*len(riskFactorNames) # Will be populated with values
    portDV01Entry = portDV01Values.copy() # Used as indicator if a RF has data
    x = min([len(i) for i in secDV01s[1]]) # Check to make sure not empty
    countsec = len(seclist) # Check to make sure same length for seclist and what's in secDV01s
    if countsec*x == 0:
        return "Error" # if OK then do work
    else:
        for i in range(countsec): # Loop over securities
            for j in range(len(secDV01s[1][i])):
                rf = secDV01s[1][i][j]
                k = riskFactorNames.index(rf) # Position in RF list
                portDV01Values[k] += secDV01s[2][i][j] # Accumulates the DV01s across securities
                portDV01Entry[k] = 1 # Set to 1 when there is data for this RF
        portDV01Names = [portDV01Names[i] for i in range(len(portDV01Entry)) if portDV01Entry[i] == 1]
        portDV01Values = [portDV01Values[i] for i in range(len(portDV01Entry)) if portDV01Entry[i] == 1]
        portDV01s = [portDV01Names, portDV01Values]
    return portDV01s


def populateEqvDV01s(seclist, settledate,df_sec,df_rf,df_market):
    seclist = df_sec[df_sec['secNames'].isin(seclist)]['secNames'].tolist()
    fullseclist = []
    secDV01Names=df_sec.loc[seclist]['secRiskFactorNames'].tolist()   
    flat_list=[item for sublist in secDV01Names for item in sublist]
    def remove_duplicates(lst): 
        seen = set() 
        result = [] 
        for item in lst: 
            if item not in seen: 
                result.append(item) 
                seen.add(item) 
        return result
    rflist = list(remove_duplicates(flat_list))    
    countsec = len(rflist)
    riskFactorNames=df_rf['riskFactorNames'].tolist()
    if countsec == 0:
        return "Error in populateEqvDV01s - seclist or rflist empty"
    else:
        for i in range(countsec):
            k = riskFactorNames.index(rflist[i])
            x = df_rf['riskFactorEqvs'][k]
            if x == "":
                continue
            else:
                fullseclist.append(x)
        holdings = [1] * len(fullseclist)
        secDV01s = populateSecDV01s(fullseclist, holdings, settledate,df_sec, df_rf, df_market)
        portDV01Names = riskFactorNames
        portDV01Values = [0] * len(riskFactorNames)
        portDV01Entry = portDV01Values.copy()
        x = min(map(len, secDV01s[1]))
        countsec = len(fullseclist)
        if countsec * x == 0:
            return "Error"
        else:
            for i in range(countsec):
                for j in range(len(secDV01s[1][i])):
                    rf = secDV01s[1][i][j]
                    k = riskFactorNames.index(rf)
                    if df_rf['riskFactorEqvs'][k] == fullseclist[i]:
                        portDV01Values[k] = secDV01s[2][i][j]
                    else:
                        continue
                    portDV01Entry[k] = 1
            portDV01Names = [name for i, name in enumerate(portDV01Names) if portDV01Entry[i] == 1]
            portDV01Values = [value for i, value in enumerate(portDV01Values) if portDV01Entry[i] == 1]
            portDV01s = [portDV01Names, portDV01Values]
        return portDV01s

##Vol, VCV, VaR functions

#VCV is Variance Covariance Matrix
#Function to pick out and populate a VCV matrix using a list of risk factors
def populateVCV(rflist):
    ##xindex=df_rf['riskFactorNames'].index(rflist)
    #xbus=sign(df_rf['riskFactorBumpUnit'][xindex])
    #xbusm=np.outer(xbus,xbus)##change sign of correlation based on whether the RF is bumped up or down
    #xVCV=VCV[xindex,xindex]##get sublist of VCV
    #np.multiple(xVCV,xbusm)
    len_rflist = len(rflist)
    vcv = np.zeros((len_rflist, len_rflist))
    vols = np.zeros(len_rflist)
    
    for i in range(len_rflist):
        k = np.where(sec_master.df_market['marketDataNames'] == rflist[i])[0][0]
        vols[i] = sec_master.marketVols[0,k]
        m = np.where(sec_master.df_rf['riskFactorNames'] == rflist[i])[0][0]##xindex
        x = np.sign(sec_master.df_rf['riskFactorBumpUnit'][m])        
        for j in range(len_rflist):
            l = np.where(sec_master.df_market['marketDataNames'] == rflist[j])[0][0]
            n = np.where(sec_master.df_rf['riskFactorNames'] == rflist[j])[0][0]
            y = x * np.sign(sec_master.df_rf['riskFactorBumpUnit'][n]) # Need to change sign of correlation based on whether the RF is bumped up or down (for yields sign is reversed) Fixed 8/2013, TSC            
            vcv[i, j] = np.multiply(sec_master.corrmatrix[k, l], y) # This line adjusts for the fact that FI is dn bump rather than up
            # vcv[i,j] = corrmatrix[k,l] # This does not adjust 
            #here, vcv is just corrlation matrix(ρ), it has not been cov    
    vols = vols.reshape(len_rflist, 1)
    vols = np.kron(vols, vols)#sd1*sd2
    vols= vols.reshape(len_rflist,len_rflist)
    vcv = np.multiply(vols,vcv)##cov1,2=rho(ρ12)*sd1*sd2
    return vcv


def secVolbyVCV(seclist, holdings, settledate,df_sec, df_rf, df_market):
    secDV01s = populateSecDV01s(seclist, holdings, settledate,df_sec, df_rf, df_market)
    length = len(secDV01s[0])
    secvol = []
    for i in range(length):
        rflist = secDV01s[1][i]
        secvcv = populateVCV(rflist)
        secvol.append(math.sqrt(np.dot(secDV01s[2][i], np.dot(secvcv, secDV01s[2][i]))))
    return [seclist, secvol]

#From 17-nov, Function to loop over RFs in a portfolio and calculate the vol of each RF
def rfVolbyVCV(seclist, holdings, settledate, df_sec, df_rf, df_market):    
    len_sec=len(seclist)
    rfDV01s = populatePortDV01s(seclist, holdings, settledate, df_sec, df_rf, df_market)
    rfvcv = populateVCV(rfDV01s[0])
    xx1 = np.kron(rfDV01s[1], rfDV01s[1])   
    xx1=xx1.reshape(len_sec, len_sec)
    xx2 = xx1 * rfvcv  # 11-jul-11 - why does this work? Needs documentation.
    rfvols = np.sqrt(np.diag(xx2))
    return [rfDV01s[0], rfvols]


#Function to calculate portfolio volatility from VCV by matrix multiplication
def portVolbyVCV(seclist, holdings, settledate, df_sec, df_rf, df_market):
    portDV01s = populatePortDV01s(seclist, holdings, settledate, df_sec, df_rf, df_market)
    rflist = portDV01s[0]
    portvcv = populateVCV(rflist)
    portvol = np.sqrt(np.dot(np.dot(portDV01s[1], portvcv), portDV01s[1]))
    return portvol

#The first part of portVolbyVCVextended is a duplicate of portVolbyVCV but then calculates and returns list which contains additional items - July 2011
 #1. portfolio volatility 
 #2. list of marginal contributions (vector) in levels (for risk factors)
 #3. marginal contributions proportional
 #4. All-or-nothing contribution (change in vol when this pos'n is set to zero)
 #5. Best hedge position (in same units as input holdings) - must fix because by RFs not securities
 #6. Volatility at best hedge
 #7. Stand-alone volatilities of individual risk factors
 #8. RF list (remember that although seclist is input, risk is done by RFs
 #9. DV01s by RFs per 1 unit holdings
 #10. Correlations of RFs with portfolio
 #11. Replicating position (= original - best hedge)
 #12. DV01s by RFs (actual holdings)
 #13. VCV matrix
def portVolbyVCVextended(seclist, holdings, settledate, df_sec, df_rf, df_market):
    # Populate portDV01s
    portDV01s = populatePortDV01s(seclist, holdings, settledate, df_sec, df_rf, df_market)
    xx1 = populateEqvDV01s(seclist, settledate,df_sec,df_rf,df_market) # RF DV01s for unit holdings of everything
    
    # Get indexes of elements with eqv DV01s into the full list of all portfolio DV01s
    indexes = [portDV01s[0].index(x) for x in xx1[0]]
    # Create a holder with the right names and zeros for DV01s
    portDV01s1 = portDV01s[:]
    portDV01s1[1] = np.zeros_like(portDV01s[1])
    
    # Populate the values of the DV01s in the right order
    portDV01s1[1][indexes] = xx1[1]
    
    rflist = portDV01s[0]
    portvcv = populateVCV(rflist)
    cov = np.dot(portvcv, portDV01s[1]) # Intermediate result - essentially covariance
    
    # Calculate portfolio variance and vol rather than variance
    x = np.dot(portDV01s[1], cov)
    portvol = np.sqrt(x)
    
    # Calculate intermediate vector, used in calculation of marginal and all-or-nothing contribution
    intermedvec = portDV01s[1] * cov
    
    # Calculate Marginal Contribution in levels and MC proportional
    mcl = intermedvec / portvol
    mcp = mcl / portvol
    
    # Get Diagonal elements - sigxx - used in a few places
    y = np.diag(portvcv)
    
    # Calculate correlation of RFs with portfolio
    corr = cov / (portvol * np.sqrt(y))
    
    # Calculate volsbyrf - All-or-nothing contribution to volatility
    volsbyrf = np.array(portDV01s[1]) * np.array(portDV01s[1]) * y
    contallornot = portvol - np.sqrt(x - 2 * intermedvec + volsbyrf)
    
    # Calculate besthedge as proportion of input holding
    besthedge = np.zeros_like(portDV01s[1])
    for i in range(len(portDV01s[1])):
        if portDV01s1[1][i] != 0:
            besthedge[i] = (portDV01s[1][i] - cov[i] / y[i]) / portDV01s1[1][i]
    
    repport = np.zeros_like(portDV01s[1])
    for i in range(len(portDV01s[1])):
        if portDV01s1[1][i] != 0:
            repport[i] = (cov[i] / y[i]) / portDV01s1[1][i]
    
    # Calculate volatility at best hedge position
    volbesthedge = np.sqrt(x - (cov * cov) / y)   
    return [portvol, mcl, mcp, contallornot, besthedge, volbesthedge, np.sqrt(volsbyrf), rflist, portDV01s1[1], corr, repport, portDV01s[1], portvcv]
 
    

#######we do not need the following functions for now, just ignore them
##Simulated P&L functions
def calcSimulatedPL(seclist, holdings, settledate, inputRFs, marketDataValues,df_sec, df_rf, df_market):
    bumpunit = 0
    countsec = len(seclist)
    countrf = len(inputRFs[0])
    rflist = inputRFs[0]
    ndraws = len(inputRFs) - 1
    if countrf * countsec == 0:
        return "Error in calcSimulatedPL - seclist or rflist empty"
    baseval = pvPort(seclist, holdings, settledate, marketDataValues)
    pandl = [[0 for i in range(countsec + 1)] for j in range(ndraws + 1)]
    pandl[0][0:countsec] = seclist
    pandl[0][countsec] = "Portfolio"
    for k in range(ndraws):
        xmarketDataup = marketDataValues.copy()
        for i in range(countrf):
            marketDataNames=df_market['marketDataNames'].tolist()
            irfmkt = marketDataNames.index(rflist[i])
            in_val = xmarketDataup['marketDataValues'][irfmkt]
            riskFactorNames=df_rf['riskFactorNames'].tolist()
            irf = riskFactorNames.index(rflist[i])
            h = inputRFs[k + 1][i]
            leveltst = df_rf['riskFactorBumpTypes'][irf]
            bumpunit = df_rf['riskFactorBumpUnit'][irf]
            if leveltst == "level":
                inup = in_val + h
                xmarketDataup[irfmkt] = inup
                #xmarketDataup.loc[irfmkt,'marketDataValues']= inup
            elif leveltst == "percent":
                inup = in_val * (1 + h)
                xmarketDataup[irfmkt] = inup
                #xmarketDataup.loc[irfmkt,'marketDataValues']= inup
            elif leveltst == "levelcurveinp":
                pass
            else:
                return "Error in calcSimulatedPL"
        ###problem: why we only have up values here in x2????    
        x2 = pvPort(seclist, holdings, settledate, xmarketDataup)
        pandl[k + 1][1:countsec + 1] = [x2[i] - baseval[i] for i in range(countsec)]
    return pandl


#we do no need  Distribution Functions for now,just ignore them
###Distribution Functions

#Define pdf, cdf, and quantile that will work with python distribtutions but also with 2-point Mixture of Normals
def pdf(dist, x):
    if not isinstance(dist, tuple) or dist[0] != 'MixtureNormals':
        return scipy.stats.norm.pdf(x, *dist)
    # If a tuple, then assumed to be {"MixtureNormals",normdist, 
    # sigma, alpha,beta}. The sigma is the SD of the mixture, 
    # and the sigma of the underlying normals are sigma/Sqrt[1-alpha+
    # alpah*beta^2] and beta*sigma...
    normdist = dist[1]
    sigma = dist[2]
    alpha = dist[3]
    beta = dist[4]
    normsig = sigma / math.sqrt((1 - alpha) + alpha * beta ** 2)
    y = (1 - alpha) * normdist.pdf(x, scale=normsig) + \
        alpha * normdist.pdf(x, scale=beta * normsig)
    return y

def cdf(dist, x):
    if not isinstance(dist, sec_master.normalmixture):
        return sec_master.CDF(dist, x)
    else:
        normdist = dist[0]
        sigma = dist[1]
        alpha = dist[2]
        beta = dist[3]
        normsig = sigma / ((1 - alpha) + alpha * beta ** 2) ** 0.5
        y = (1 - alpha) * normdist.cdf(normsig, x) + alpha * normdist.cdf(beta * normsig, x)
        return y


def quantile(dist, p):
    if not isinstance(dist, tuple) or dist[0] != 'MixtureNormals':
        return norm.ppf(p)
    else:
        sigma = dist[2]
        alpha = dist[3]
        beta = dist[4]
        normsig = sigma / ((1 - alpha) + alpha * beta**2)**0.5
        xstart = norm.ppf(p, scale=sigma)
        y = optimize.newton(lambda x: cdf(dist, x) - p, xstart)
        return y    
#method2:
def mixture_params(dist):
    if isinstance(dist, list) and dist[0] == "MixtureNormals":
        normdist = dist[1]
        sigma = dist[2]
        alpha = dist[3]
        beta = dist[4]
        normsig = sigma / math.sqrt((1 - alpha) + alpha * beta**2)
        return (normdist, sigma, alpha, beta, normsig)
    else:
        return None

def pdf(dist, x):
    params = mixture_params(dist)
    if params is None:
        return stats.norm.pdf(x, *dist)
    else:
        normdist, sigma, alpha, beta, normsig = params
        y = (1 - alpha) * stats.norm.pdf(x, normdist(loc=0, scale=normsig)) \
            + alpha * stats.norm.pdf(x, normdist(loc=0, scale=beta*normsig))
        return y

def cdf(dist, x):
    params = mixture_params(dist)
    if params is None:
        return stats.norm.cdf(x, *dist)
    else:
        normdist, sigma, alpha, beta, normsig = params
        y = (1 - alpha) * stats.norm.cdf(x, normdist(loc=0, scale=normsig)) \
            + alpha * stats.norm.cdf(x, normdist(loc=0, scale=beta*normsig))
        return y

def quantile(dist, p):
    params = mixture_params(dist)
    if params is None:
        return stats.norm.ppf(p, *dist)
    else:
        normdist, sigma, alpha, beta, normsig = params
        xstart = stats.norm.ppf(p, loc=0, scale=sigma)
        y = optimize.newton(lambda x: cdf(dist, x) - p, xstart)
        return y
    
    
# Define VaR as the quantile      
def var(dist, prob, vol):
    return quantile(dist(vol), prob)

    
#Define some other distribution functions, each takes a parameter "vol" which is the std dev
def normal(vol):
    return norm(loc=0, scale=vol)

studentdof = 6
def student(vol):
    return t(df=studentdof, loc=0, scale=vol * np.sqrt((studentdof - 2) / studentdof))

alpha = 0.05
beta = 3
def normalmix(vol):
    normdist = normal(vol)
    sigma = vol
    return ["normalmixture", normdist, sigma, alpha, beta]

###From here, we need the following functions to generate the tables in CH10 in textbook##

###Supplemental Document 4 - Simple VaR Calculator
##Initialize Data and Functions

# Set the desired locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

##Table Functions for CH10 tables############

#Table that produces Pos'n Vol, MC Levels, All-or-nothing Cont'n, Best Hedges Pos'n, Vol at Best Hedges, for arbitrary no. of instruments. Problem when there is not a 1-to-1 correspondence between securities and RFs - in such a case the function does not put amounts in the labels.
def tableContributers(ordering, rflist, portvol, volexplained, combvol, nav):
    k = len(ordering)
    xx2 = [
        ["", "Exp Vol", "", "Curr Pos'n", "Trade to best", "% Reduction in Vol to", "..."],
        ["", "(1-sig P&L)", "Contribution", "(mn eqv)", "Hedge (eqv)", "Best Hedge", "Zero Pos'n"]
    ]
    xx3 = xx2 + [
        [
            rflist[ordering[i]],
            f"{(100 * portvol[6][ordering[i]] / nav):.4f}" if isinstance(portvol[6][ordering[i]], float) and nav > 1 else f"{portvol[6][ordering[i]]}",
            f"{(100 * portvol[2][ordering[i]]):.1f}",
            f"{(portvol[4][ordering[i]] + portvol[10][ordering[i]]):.1f}",
            f"{(-portvol[10][ordering[i]]):.1f}",
            f"{volexplained[ordering[i]]:.1f}",
            f"{(100 * portvol[3][ordering[i]] / combvol):.1f}"
        ]
        for i in range(k)
    ]
    return xx3

def reportFunction1(seclist, holdings, settledate, nav):
    xx, portvol, yy, x, y, combvol, i, table, rflist, xx2, xx3, xx4, xx5, xx6, zz, volexplained, ordering, replicateport = [None] * 18
    portvol = portVolbyVCVextended(seclist, holdings, settledate,sec_master.df_sec, sec_master.df_rf, sec_master.df_market)
    combvol = portvol[0]
    rflist = portvol[7]
    volexplained = 100.0 * (1 - portvol[5] / combvol)
    yy = ["FI - rates", "FI - swap spreads", "Credit", "Equity", "Commodity", "FX", "Volatility"]
    
    xx = [sec_master.df_rf['riskFactorCategory'][i] for i in rflist]
    xdv01s = np.array(portvol[11])
    xvcv = np.array(portvol[12])
    zz = np.zeros(len(yy))
    
    for i in range(len(yy)):
        #xx4 = [0] * len(xx)
        #xx4 = np.zeros(xx)
        xx3 = [j for j in range(len(xx)) if xx[j] == yy[i]]
        zz[i] = np.sqrt( np.matmul(np.matmul(xdv01s[xx3],xvcv[xx3,:][:,xx3]),xdv01s[xx3]) )


    if nav > 1:
        xx0 = [[
            yy[i],
            f"{(100 * zz[i] / float(nav)):.4f}",
            f"{(100 * sum(portvol[2][k] for k in range(len(xx)) if xx[k] == yy[i])):.1f}",
            f"{((sum(portvol[2][k] for k in range(len(xx)) if xx[k] == yy[i])) * combvol / float(zz[i])):.3f}" if zz[i] > 0 else ""
        ] for i in range(len(yy))]
        table = [
            ["", "Exp Vol (%)", "Contribution", "Correlation"],
            ["Overall", f"{(100 * combvol / float(nav)):.4f}", "100.0", "with portfolio"]
        ]
    else:
        xx0 = [[
            yy[i],
            f"{int(zz[i]):d}"if not np.isnan(zz[i]) else "",
            f"{(100 * sum(portvol[2][k] for k in range(len(xx)) if xx[k] == yy[i])):.1f}",
            f"{((sum(portvol[2][k] for k in range(len(xx)) if xx[k] == yy[i])) * combvol / zz[i]):.3f}" if zz[i] > 0 else ""
        ] for i in range(len(yy))]
        table = [
            ["", "Exp Vol ($)", "Contribution", "Correlation"],
            ["Overall", f"{int(combvol):d}", "100.0", "with portfolio"]
        ]

    table += xx0
    ordering = sorted(range(len(portvol[2])), key=lambda i: -portvol[2][i])[:3]
    xx1 = tableContributers(ordering, rflist, portvol, volexplained, combvol, nav)
    ordering = sorted(range(len(portvol[2])), key=lambda i: portvol[2][i])[:1]
    xx2 = tableContributers(ordering, rflist, portvol, volexplained, combvol, nav)
    ordering = sorted(range(len(portvol[5])), key=lambda i: portvol[5][i])[:3]
    xx3 = tableContributers(ordering, rflist, portvol, volexplained, combvol, nav)
    replicateport = bestReplicatePort(portvol[11], portvol[12], portvol[7], portvol[8])
    xx4 = [["", "One Asset", "One Asset", "Three Assets", "Three Assets", "Five Assets", "Five Assets"], 
       ["", "% Var", "% Vol", "% Var", "% Vol", "% Var", "% Vol"],
        ["%Var / %Vol Explained"] + list(np.concatenate([
            [f"{(100 * replicateport[2][i]):.1f}", f"{(100 * replicateport[3][i]):.1f}"] for i in [0, 2, 4]
        ])),
        ["", "Asset", "Eqv Pos'n", "Asset", "Eqv Pos'n", "Asset", "Eqv Pos'n"],
        ["Asset / Eqv Pos'n"] + list(np.concatenate([
            [replicateport[4][0], f"{replicateport[6][i][0]:.1f}"] for i in [0, 2, 4]
        ])),
        ["", "", ""] + list(np.concatenate([
            [replicateport[4][1], f"{replicateport[6][i][1]:.1f}"] for i in [2, 4]
        ])),
        ["", "", ""] + list(np.concatenate([
            [replicateport[4][2], f"{replicateport[6][i][2]:.1f}"] for i in [2, 4]
        ])),
        ["", "", "", "", "", replicateport[4][3], f"{replicateport[6][4][3]:.1f}"],
        ["", "", "", "", "", replicateport[4][4], f"{replicateport[6][4][4]:.1f}"]
    ]
    table = [table, xx1, xx2, xx3, xx4]
    return table

def volbySubPortfolio(seclist, holdings, settledate, subportfolio):
    subportlist = list(set(subportfolio))  # List of unique asset classes
    noclass = len(subportlist)  # Number of asset classes
    zeros = np.zeros(len(holdings))  # Used for holding subportfolio holdings
    results = portVolbyVCVextended(seclist, holdings, settledate,sec_master.df_sec, sec_master.df_rf, sec_master.df_market)  # For complete portfolio
    
    for i in range(noclass):
        x1 = [j for j in range(len(subportfolio)) if subportfolio[j] == subportlist[i]]  # Indexes of this asset class in seclist
        xhold = zeros.copy()
        for j in x1:
            xhold[j] = holdings[j]  # Insert holdings for this subportfolio
        x2 = portVolbyVCVextended(seclist, xhold, settledate, sec_master.df_sec, sec_master.df_rf, sec_master.df_market)
        results = np.concatenate((results, x2), axis=0, dtype=object)

    partial = np.dot(np.array(results[11]), results[12])  # Matrix multiplication of VCV . Delta
    
    table = [["", "Exp Vol ($)", "Contribution", "Correlation"],
             ["Overall", "{:.0f}".format(results[0]), "100.0", "with portfolio"]]
    
    for i in range(noclass):
        x2 = results[13*(i+1)]
        x3 = 100 * np.dot(results[13*(i+1) + 11], partial) / (results[0]**2)
        if x2 > 0:
            contribution = "{:.3f}".format(x3 * results[0] / (100 * x2))
        else:
            contribution = ""
        x2_formatted = np.round(x2, decimals=0)
        x3_formatted = np.round(x3, decimals=1)
        row = [subportlist[i], x2_formatted, x3_formatted, contribution]
        table.append(row)
    
    return table




def varsumm(seclist, holdings, settledate):
    x1 = portVolbyVCV(seclist, holdings, settledate,sec_master.df_sec, sec_master.df_rf, sec_master.df_market)
    alpha = 0.01
    beta = 5

    def normalmix(vol):
        mixfact = math.sqrt(1 - alpha + alpha * beta * beta)
        normal1 = norm(loc=0, scale=vol / mixfact)
        normal2 = norm(loc=0, scale=beta * vol / mixfact)
        return (1 - alpha) * normal1.ppf(1 / 255) + alpha * normal2.ppf(1 / 255)

    x2 = [
        ["VOLATILITY and 1-ot-of 255 VaR", None],
        ["Volatility", x1],
        ["VaR Normal", norm.ppf(1 / 255, loc=0, scale=x1)],
        ["VaR Student-t (6df)", x1 * math.sqrt(4 / 6) * t.ppf(1 / 255, df=6)],
        ["VaR Normal Mix (alpha=1% beta=5)", normalmix(x1)],
        ["VaR 4-sigma rule-of-thumb", -4 * x1]
    ]

    return x2

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def numberForm(expr, n, f=None, digit_block=3, number_point=""):
    if isinstance(expr, str):
        return expr  # Return the input string as it is
        
    if f is not None:
        formatted_expr = "{:0.{}f}".format(expr, f)
    else:
        formatted_expr = str(expr)

    if '.' in formatted_expr:
        integral_part, fractional_part = formatted_expr.split('.')
        if len(integral_part) > n:
            return "{:0.{}e}".format(expr, n-1)
        elif len(integral_part) + 1 + len(fractional_part) > n:
            return "{:0.{}f}".format(expr, n - len(integral_part) - 1)
    
    formatted_expr = formatted_expr.replace('.', number_point)
    formatted_expr = ','.join(formatted_expr.rsplit(',', digit_block))
    return formatted_expr


def detailtable(seclist, holdings, settledate):
    table = [["Yield Curve", "USD", "GBP", "EUR"]]
    table2 = [["Yield Curve", "USD", "GBP", "EUR"]]
    portvol = portVolbyVCVextended(seclist, holdings, settledate)
    datamc = [["" for _ in range(3)] for _ in range(5)]
    datavol = [["" for _ in range(3)] for _ in range(5)]
    currlist = ["USD", "GBP", "EUR"]
    matlist = ["Yld2yr", "Yld5yr", "Yld10yr", "Yld30yr", "FX"]

    for i in range(5):
        for j in range(3):
            k = [idx for idx, val in enumerate(portvol[8]) if val == currlist[j] + matlist[i]]
            if len(k) == 1:
                k = k[0]
                datamc[i][j] = 100 * portvol[3][k]
                datavol[i][j] = sign(portvol[12][k]) * portvol[7][k]

    for i in range(5):
        table.append([matlist[i]] + [numberForm(val, 3, 1) for val in datamc[i]])
        table2.append([matlist[i]] + [numberForm(val, 4, 0, 3) for val in datavol[i]])

    return [table, table2]



def displayRF(seclist, holdings, settledate):
    secDV01s = populateSecDV01s(seclist, holdings, settledate,sec_master.df_sec,sec_master.df_rf,sec_master.df_market)
    secVols = secVolbyVCV(seclist, holdings, settledate,sec_master.df_sec, sec_master.df_rf, sec_master.df_market)
    rfVols = rfVolbyVCV(seclist, holdings, settledate,sec_master.df_sec, sec_master.df_rf, sec_master.df_market)
    outTable = []
    header = ["Security", "Notional", "Security Vol", "Risk Factor", "DV01", "DV01 units", "RF unit Vol", "RF pos'n Vol"]
    outTable.append(header)
    for i in range(len(seclist)):
        x = numbers.get_currency_symbol(seclist[i], 'en_US')
        x = x[0] + str(holdings[i]) + "mn"
        y = numberForm(secVols[1][i], 4, 0, digit_block=3, number_point="")
        security_row = [seclist[i], x, y, "", "", "", "", ""] 

        for j in range(len(secDV01s[1][i])):
            x = ["", "", "", secDV01s[1][j][0], numberForm(secDV01s[2][j][0], 5, 0, digit_block=3, number_point=""), "", "", ""]
            riskFactorBumpTypes=sec_master.df_rf['riskFactorBumpTypes']
            riskFactorNames=sec_master.df_rf['riskFactorNames'].tolist()
            y = riskFactorBumpTypes[riskFactorNames.index(x[3])]
            x[5] = y
            marketDataNames=sec_master.df_market['marketDataNames'].tolist()
            z = sec_master.marketVols[0,marketDataNames.index(x[3])]                    
            xx = numberForm(z, 5, 5 if y == "percent" else 4,3)
            x[6] = xx
            y = numberForm(abs(secDV01s[2][j][0] * z), 4, 0)
            x[7] = y

            combined_row = security_row[:3] + x[3:]  
            outTable.append(combined_row)
    table_str = tabulate(outTable, headers="firstrow", tablefmt="pipe")    
    return table_str

##page 328,335 in the book
def bestReplicatePort(delta, sigma, rfnames, rfunitDV01s):
    nosec = 5  # Build replicating portfolios for 1-5 securities
    portsize = len(delta)
    part = np.dot(sigma, delta)
    portvar = np.dot(delta, part)
    index = np.zeros(nosec)
    index = index - 1
    dstarint5 = [[]]  # These will be a ragged array of positions, names, etc.
    names = []
    amounts = [[]]
    portvar5 = np.zeros(nosec)
    k=None
    for i in range(nosec):
        varchk = portvar
        for j in range(portsize):  # Loop through assets
            if j in index:
                continue  # If this asset has already been selected, skip
            index[i] = j   # Add this asset's index to the list of selected assets
            partint = part[index.astype(int)[:i+1]]# Select appropriate assets
            sigmaint = sigma[index.astype(int)[:i+1],:][:,index.astype(int)[:i+1]]       
            dstarint = solve(sigmaint, partint)  # Intermediate result
            varint = portvar - np.dot(partint, dstarint)  # Variance for this set of assets
            if varint < varchk:  # If lowest so far
                varchk = varint
                k = j # Keep track of index
                
        
        index[i]=k # Now do everything for the best of this loop         
        partint = part[index.astype(int)[:i+1]]
        sigmaint = sigma[index.astype(int)[:i+1],:][:,index.astype(int)[:i+1]]
        portvar5[i] = varchk
        dstarint5.append(solve(sigmaint, partint))  # This is the position at best "best hedge"
        names.append(rfnames[k])
        y = []
        for j in range(i+1):
            if rfunitDV01s[index.astype(int)[j] - 1] != 0:
                x = dstarint5[i+1][j] / rfunitDV01s[index.astype(int)[j]]
            else:
                x = 0
            y.append(x)
        amounts.append(y)

    portvarPercent = 1 - (portvar5 / portvar)  # Percentage of original portfolio variance explained
    portvolPercent = 1 - (np.sqrt(portvar5) / np.sqrt(portvar))  # Percentage vol explained
    dstarint5 = dstarint5[1:6]  # Drop off the first element (empty)
    amounts = amounts[1:6]    
    return (np.sqrt(portvar), np.sqrt(portvar5), portvarPercent, portvolPercent, names, dstarint5, amounts)



