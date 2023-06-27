# RiskManagement-Public

## Introduction

Based on the methodology in _Quantiative Risk Management_ by Thomas Coleman (Wiley, 2012)

First draft of the python code to implement the risk reports and example portfolio discussed in Section 10.6 "Risk Reporting". Code by Thomas Coleman, Kunyi Du, Dana Shen.

## Table_Ch10.py

File that runs the risk report tables for Section 10.6


## Example Portfolio
+ Government subportfolio
  + Long $20mn US Treasury 10-year bond
  + Lond £25mn U.K. Gile 10-year
  + Short $20mn notional call option on a 5-year US Treasury
+ Swaps subportfolio
  + Shor $20mn 10-year swap
  + Long outright $30mn US Treasury 
  + Net is long swap spreads and long some resudual US Treasury exposure
+ Credit subportfoli
  + Long £55mn corporate bond spread (credit default swap or CDS on France Telecom)
+ Equity subportoflio
  + Long eur7mn CAC futures
  + Long eur5mn French company (France Telecom)


## SamplePort_define.py

Defines the dataframes for the example portfolio
+ df_sec - Defines the securities, basically the "Securities Master"
  + secNames, secType, secParmNames, secParmValues, secMktInputNames, secRiskFactorNames, secPVFunction
+ df_rf - Defines risk factors
  + riskFactorNames, riskFactorCategory, riskFactorEqvs, riskFactorBumpSize, riskFactorBumpTypes, riskFactorBumpUnit
+ df-market - market data
+ corrmatrix and marketVols - the elements of the variance-covariance matrix for historical risk factors, as of a date in 2011

## RiskMgmtFunctions.py

All the functions for calculating and displaying risk measures. 

+ PV functions for each security - defined as PV[inputs, settledate, holding, parms]. input is things like yield, forward curve, FX rates, etc - the market data. holding is the holding. parms is parameters such as coupon or maturity date.
  +   bondpv, bondoptionpv, swappv, eqtyFutpv, simpCDSpv, eqtypv, cashpv, swappvactual

+ dv01 functions
  + dv01posn - Calculate DV01 (sensitivity) to a risk factor
  + populateSecDV01s - Populates a 3 x n array with the first row being the list of securities, the second row the list of risk factors (maybe multiple per security), and the third the values.
  + populatePortDV01s - Populates a 2 x n array with the first row being the list of risk factors and the second row the values of the DV01s. 
  + populateEqvDV01s - Populates the same array with populatePortDV01s, but for a unit holding of the equivalent security in its native currency.

+ Utility functions
  + fxrateSec - pick out FX rate of security
  + populateVCV - Pick out and populate a VCV matrix using a list of risk factors

+ Volatility functions
  + secVolbyVCV - Loop over the securities in a portfolio and calculate the vol of each security
  + rfVolbyVCV -  Loop over RFs in a portfolio and calculate the vol of each RF
  + portVolbyVCV -  Calculate portfolio volatility from VCV by matrix multiplication
  + portVolbyVCVextended 
    + the most important function, calculates all statistics such as marginal contribution, here is the output:
      + portfolio volatility 
      + list of marginal contributions (vector) in levels (for risk factors)
      + marginal contributions proportional
      + All-or-nothing contribution (change in vol when this pos'n is set to zero)
      + Best hedge position (in same units as input holdings) - must fix because by RFs not securities
      + Volatility at best hedge
      + Stand-alone volatilities of individual risk factors
      + RF list (remember that although seclist is input, risk is done by RFs
      + DV01s by RFs per 1 unit holdings
      + Correlations of RFs with portfolio
      + Replicating position (= original - best hedge)
      + DV01s by RFs (actual holdings)
      + VCV matrix 
  + volbySubPortfolio - Expected Volatility by Subportfolio
  + bestReplicatePort - Build replicating portfolios for 1-5 securities

+ Table display functions
  + reportFunction1 - Creates 5 tables, returned in a list (remember tha python indexes from 0):
    + Summary expected volatility (total and by asset class)
    + Top 3 positive contributors to risk
    + Top 1 negative contributor to risk
    + Top 3 best hedges
    + Best replicating portfolio
  + varsumm - VOLATILITY and 1-out-of 255 VaR 
  + displayRF - Display the securities and RFs, DV01s, and Vols
