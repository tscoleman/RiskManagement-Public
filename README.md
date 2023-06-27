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
+ Full list to be added
