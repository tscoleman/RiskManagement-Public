import RiskMgmtFunctions as sec_master
import SamplePort_define  as fns
#%% Start testing date functions

#print(contents)
###Dates

#serialday function
#input of serialdat is date need to be datetime()
#serialday output is int
settledate = fns.datetime(2009, 1, 27, 0, 0, 0)
se=fns.serialday(settledate)
print(settledate)
type(se)

#dateday function:input is int
#dateday output is datetime.datetime
dd=fns.dateday(39842)
print(dd)
type(dd)

#%% Testing PV functions


#pv function
pv = fns.pv(rate=0.0258, nper=2, pmt=3.75, fv=100)
pv

#bondprcsimp function
security = '5yr UST'
parms = sec_master.df_sec.loc[security]['secParmValues']
yld = sec_master.df_sec.loc[security]['InputsValue'][0] 
holding = sec_master.df_sec.loc[security]['SecHolding'][0] 
bond_prcsimp=fns.bondprcSimp(yld, settledate, parms)
bond_prcsimp

#bondprc fuction
security = '5yr UST'
parms = sec_master.df_sec.loc[security]['secParmValues']
yld = sec_master.df_sec.loc[security]['InputsValue'][0] 
holding = sec_master.df_sec.loc[security]['SecHolding'][0] 
bond_prc=fns.bondprc(yld, settledate, parms)
bond_prc

#bondpv function
security = '5yr UST'
parms = sec_master.df_sec.loc[security]['secParmValues']
holding = sec_master.df_sec.loc[security]['SecHolding'][0] 
inputs = sec_master.df_sec.loc[security]['InputsValue']
bond_pv=fns.bondpv(inputs, settledate, holding, parms)
bond_pv


#Swap Functions (both generic and actual swap)
#actual swap function
security = "Swap201901"
parms = sec_master.df_sec.loc[security]['secParmValues']
inputs = sec_master.df_sec.loc[security]['InputsValue']
swappvactual=fns.swappvactual(inputs, settledate, holding, parms)
swappvactual

###swappvfunction 
#UST yield change every minute since economy change,0.01=100bp company spread not change much;
#spread change because of company risk change.
#coupon rate,parms[0]=0.028<0.0358=0.0258+0.01 means that Coupon<Yrisky(market yield),discount bond,which means value=pv of the swap should be negative, lose money
## a generic 10-yr swap, which uses the "swappv" function
security = '10yr Swap'
inputs = sec_master.df_sec.loc[security]['InputsValue']
parms = sec_master.df_sec.loc[security]['secParmValues']
swappv=fns.swappv(inputs, settledate, holding, parms)
swappv


###Bond option function
security = "5yr Bond Opt"
#Expiry=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][0]

#bondoptionprc_inputs[Expiry,rshort,Forward yield,Strike yield,Volatility,Underlier Term,Underlier Frequency,P=put = payers option]
parms = sec_master.df_sec.loc[security]['secParmValues']
expiry=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][0]
rshort=sec_master.df_sec.loc['5yr Bond Opt']['InputsValue'][0]
fyld=sec_master.df_sec.loc['5yr Bond Opt']['InputsValue'][1]
strike=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][1]
vol=sec_master.df_sec.loc['5yr Bond Opt']['InputsValue'][2]
underlierTerm=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][2]
underlierFreq=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][3]
putcall=sec_master.df_sec.loc['5yr Bond Opt']['secParmValues'][4]
bondoptionprc=fns.bondoptionprc(expiry, rshort,fyld, strike, vol, underlierTerm, underlierFreq, putcall)
bondoptionprc

security = '5yr Bond Opt'
inputs = sec_master.df_sec.loc[security]['InputsValue']
parms = sec_master.df_sec.loc[security]['secParmValues']
holding = sec_master.df_sec.loc[security]['SecHolding'][0] 
bondoptionpv=fns.bondoptionpv(inputs, settledate, holding, parms)
bondoptionpv


security = "FTE CDS"
parms = sec_master.df_sec.loc[security]['secParmValues']
inputs = sec_master.df_sec.loc[security]['InputsValue']
simpCDSpv=fns.simpCDSpv(inputs, settledate, holding, parms)
simpCDSpv

security = "EURCash"
parms = sec_master.df_sec.loc[security]['secParmValues']
inputs = sec_master.df_sec.loc[security]['InputsValue']
cashpv=fns.cashpv(inputs, settledate, holding, parms)
cashpv


security =  "FTE Equity"
parms = sec_master.df_sec.loc[security]['secParmValues']
inputs = sec_master.df_sec.loc[security]['InputsValue']
eqtypv=fns.eqtypv(inputs, settledate, holding, parms)
eqtypv

security = "SPX Index Futures"
parms = sec_master.df_sec.loc[security]['secParmValues']
inputs = sec_master.df_sec.loc[security]['InputsValue']
eqtyFutpv=fns.eqtyFutpv(inputs, settledate, holding, parms)
eqtyFutpv


security = "SPX Index Futures"
parms = sec_master.df_sec.loc[security]['secParmValues']
inputs = sec_master.df_sec.loc[security]['InputsValue']
futpv=fns.futpv(inputs, settledate, holding, parms)
futpv

#%% DV01 functions
riskfactor = 'USDYld5yr'
security = '5yr UST'
dv01posn = fns.dv01posn(security, settledate, riskfactor, holding)
dv01posn

security = '5yr UST'
marketDataValues = sec_master.df_market
pvposn=fns.pvposn(security, holding, settledate, marketDataValues)
pvposn



security = '5yr Bond Opt'
marketDataValues = sec_master.df_market
pvposn=fns.pvposn(security, holding, settledate, marketDataValues)
pvposn

seclist = ['5yr UST','10yr UST']
holdings=[1,2]
pvPort=fns.pvPort(seclist, holdings, settledate, marketDataValues)
pvPort

dv01sec=fns.dv01sec(security, settledate, riskfactor)
dv01sec

#Security list DV01 functions
seclist = ['5yr UST','10yr UST']
holdings=[1,1]
df_sec=sec_master.df_sec
df_market=sec_master.df_market
df_rf=sec_master.df_rf
populateSecDV01s=fns.populateSecDV01s(seclist, holdings, settledate, df_sec,df_rf,df_market)
populateSecDV01s

###########problem in "5yr Bond Opt"
seclist = ["10yr UST", "5yr Bond Opt", "10yr UST", "10yr Swap", "10yr UKG", "FTE CDS", "CAC Index Futures", "FTE Equity", "GBPCash", "EURCash", "5yr UST"]
holdings=[20, -20, 30, -20, 25, 55, 7, 5, -10, 0, 0]
df_sec=sec_master.df_sec
df_market=sec_master.df_market
df_rf=sec_master.df_rf
populateSecDV01s=fns.populateSecDV01s(seclist, holdings, settledate, df_sec,df_rf,df_market)
populateSecDV01s


populatePortDV01s=fns.populatePortDV01s(seclist, holdings, settledate,df_sec, df_rf, df_market)
populatePortDV01s

seclist = ['5yr UST','10yr UST']
holdings=[1,2]
df_sec=sec_master.df_sec
df_market=sec_master.df_market
df_rf=sec_master.df_rf
populateEqvDV01s=fns.populateEqvDV01s(seclist, settledate,df_sec,df_rf,df_market)
populateEqvDV01s

#%% Vol, VCV, VaR functions
rflist = ['USDYld5yr','USDYld10yr']
populatevcv = fns.populateVCV(rflist)
populatevcv

 
seclist = ['5yr UST','10yr UST']
holdings=[1,2]
secVolbyVCV = fns.secVolbyVCV(seclist, holdings, settledate,df_sec, df_rf, df_market)
secVolbyVCV

seclist = ['5yr UST','10yr UST']
holdings=[1,2]
df_sec=sec_master.df_sec
df_market=sec_master.df_market
df_rf=sec_master.df_rf
rfVolbyVCV=fns.rfVolbyVCV(seclist, holdings, settledate, df_sec, df_rf, df_market)
rfVolbyVCV

seclist = ['5yr UST','10yr UST']
holdings=[1,2]
portVolbyVCV=fns.portVolbyVCV(seclist, holdings, settledate, df_sec, df_rf, df_market)
portVolbyVCV

seclist = ['5yr UST','10yr UST']
holdings=[1,2]
portVolbyVCVextended=fns.portVolbyVCVextended(seclist, holdings, settledate, df_sec, df_rf, df_market)
portVolbyVCVextended

seclist = ["10yr UST", "5yr Bond Opt", "10yr UST", "10yr Swap", "10yr UKG", "FTE CDS", "CAC Index Futures", "FTE Equity", "GBPCash", "EURCash", "5yr UST"]
holdings=[20, -20, 30, -20, 25, 55, 7, 5, -10, 0, 0]
portVolbyVCVextended=fns.portVolbyVCVextended(seclist, holdings, settledate, df_sec, df_rf, df_market)
portVolbyVCVextended
###ignore the following
seclist = ['5yr UST','10yr UST']
holdings=[1,2]
df_sec=sec_master.df_sec
df_market=sec_master.df_market
df_rf=sec_master.df_rf
inputRFs = [["USDYld5yr", "USDYld10yr"],
  [0.0148, 0.0248],
  [0.014, 0.02],
  [0.013, 0.0148]]
#calcSimulatedPL=fns.calcSimulatedPL(seclist, holdings, settledate, inputRFs, marketDataValues,df_sec, df_rf, df_market)
#calcSimulatedPL


##utility function will be used in ch10 table
sec = '5yr UST'
df_sec=sec_master.df_sec
df_market=sec_master.df_market
fxrateSec=fns.fxrateSec(sec, df_sec, df_market)
fxrateSec








