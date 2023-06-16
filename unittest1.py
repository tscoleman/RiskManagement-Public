## file_path = "C:\\Users\\SJY\\Documents\\GitHub\\RiskManagement\\SamplePort2_python_define.py"
#file_path = "/Users/mori/Documents/GitHub/RiskManagement/SamplePort2_python_define.py"
##file_path = "/Users/tcoleman/tom/Economics/Harris/research/RiskManagement/pythoncode/SamplePort2_python_define.py"

## file_path = "C:\\Users\\SJY\\Documents\\GitHub\\RiskManagement\\DistFn1_python_function.py"
#file_path = "/Users/mori/Documents/GitHub/RiskManagement/DistFn1_python_function.py"
##file_path = "/Users/tcoleman/tom/Economics/Harris/research/RiskManagement/pythoncode/DistFn1_python_function.py"
   
## file_path = "C:\\Users\\SJY\\Documents\\GitHub\\RiskManagement\\SamplePort2_python_define.py"
#file_path = "/Users/mori/Documents/GitHub/RiskManagement/SamplePort2_python_define.py"

import unittest
from datetime import datetime, timedelta

import SamplePort_df_python_define
import DistFn_df_python_function  

#%% Start testing date functions


#serialday function
#input of serialdate is date need to be datetime()
#serialday output is int. 
#  - Number of days since 30-dec-1899
#  - This matches the Windows Excel format 
#    - except for the days between 30-dec-1899 and 1-mar-1900, because Windows Excel inherited the 1900 leap-year error from Lotus 1-2-3
#  - the date 27-jan-2009 is 39840 days since 30-dec-1899 
#  - So the following checks that the date functions produce 39840, and that goes back to 27-jan-2009

settledate = datetime(2009, 1, 27, 0, 0, 0)

class TestDates(unittest.TestCase):
    def test_serialday(self):
        self.assertEqual(DistFn_df_python_function.serialday(settledate), 39840)
##    def test_serialday2(self):
##        self.assertEqual(DistFn1_python_function.serialday(datetime(2009, 1, 27, 0, 0, 0)), 39842)
    #dateday function:input is int
    #dateday output is datetime.datetime
    def test_dateday(self):
        self.assertEqual(DistFn_df_python_function.dateday(39840), settledate)


##unittest.main(2)   ## 0 = quiet, 1=default, 2=verbose

if __name__ == '__main__':
    unittest.main()

#%%
class TestFunction(unittest.TestCase):
    def test_pv(self):
        rate=0.0258
        nper=2
        pmt=3.75
        fv=100
        self.assertAlmostEqual(DistFn_df_python_function.pv(rate, nper, pmt, fv), 102.252, places=3)
     
    def test_bondprcSimp(self):
        yld = 0.0158
        settledate = datetime(2009, 1, 27)
        parms = [3.75, [2013, 11, 15], 2]   # Semi-annual bond
        self.assertAlmostEqual(DistFn_df_python_function.bondprcSimp(yld, settledate, parms), 110.393, places=3)

    def test_bondprc(self):
        yld = 0.0158
        settledate = datetime(2009, 1, 27)
        parms = [3.75, [2013, 11, 15], 2]   # Semi-annual bond
        result = DistFn_df_python_function.bondprc(yld, settledate, parms)
        if isinstance(result, list):
            result = result[0]
        self.assertAlmostEqual(result, 109.988, places=3)
        

    def test_bondpv(self):
        inputs = [0.0158, 1]
        settledate = datetime(2009, 1, 27)
        holding = 1
        parms = [3.75, [2013, 11, 15], 2]
        result = DistFn_df_python_function.bondpv(inputs, settledate, holding, parms)
        expected_result = 1.10743998 * (10 ** 6)
        self.assertAlmostEqual(result, expected_result, places=2)
    def test_swappvactual(self):
        inputs = [0.0258, 0.01, 1]
        settledate = datetime(2009, 1, 27)
        holding = 1
        parms = [2.8, [2019, 1, 15, 0, 0, 0], 2, 2.3, 'USD']
        result = DistFn_df_python_function.swappvactual(inputs, settledate, holding, parms)
        expected_result = -58863.8
        self.assertAlmostEqual(result, expected_result, places=1)    
    def test_swappv(self):
        inputs = [0.0258, 0.01, 1]
        settledate = datetime(2009, 1, 27)
        holding = 1
        parms = [2.8, 10, 2, 'USD']
        result = DistFn_df_python_function.swappv(inputs, settledate, holding, parms)
        expected_result = -65082
        self.assertAlmostEqual(result, expected_result, places=0)  
    def test_bondoptionprc(self):
        expiry=0.5
        rshort=0.02
        fyld=0.05
        strike=0.0168
        vol=0.2
        underlierTerm=5
        underlierFreq=2
        putcall='C'    
        result = DistFn_df_python_function.bondoptionprc(expiry, rshort,fyld, strike, vol, underlierTerm, underlierFreq, putcall)
        expected_result = 1.42384*(10**(-15))
        self.assertAlmostEqual(result, expected_result, places=2)   
    def test_bondoptionpv(self):
        inputs=[0.02, 0.05, 0.2, 1]
        settledate = datetime(2009, 1, 27)
        holding=1
        parms=[0.5, 0.0168, 5, 2, 'C', 'USD']
        result = DistFn_df_python_function.bondoptionpv(inputs, settledate, holding, parms)
        expected_result = 1.42384*(10**(-15))
        self.assertAlmostEqual(result, expected_result, places=2)
        
    def test_simpCDSpv(self):
        inputs=[0.0258, 0.01, 0.7692307692307692]
        settledate = datetime(2009, 1, 27)
        holding=1
        parms=[1.5, 5, 2, 'GBP']
        result = DistFn_df_python_function.simpCDSpv(inputs, settledate, holding, parms)
        expected_result = 29516.8
        self.assertAlmostEqual(result, expected_result, places=1)
    def test_cashpv(self):
        inputs=[1.11]
        settledate = datetime(2009, 1, 27)
        holding=1
        parms=['EUR']
        result = DistFn_df_python_function.cashpv(inputs, settledate, holding, parms)
        expected_result = 900901
        self.assertAlmostEqual(result, expected_result, places=0)  
    def test_eqtypv(self):
        inputs=[1, 0.7692307692307692]
        settledate = datetime(2009, 1, 27)
        holding=1
        parms=[0.7, 1, 'EUR']
        result = DistFn_df_python_function.eqtypv(inputs, settledate, holding, parms)
        expected_result = 1*(10**6)
        self.assertAlmostEqual(result, expected_result, places=0)   

    def test_eqtyFutpv(self):
        inputs=[4000, 1]
        settledate = datetime(2009, 1, 27)
        holding=1
        parms=[1, "USD"]
        result = DistFn_df_python_function.eqtyFutpv(inputs, settledate, holding, parms)
        expected_result = 4000000000
        self.assertAlmostEqual(result, expected_result, places=0) 
    def test_futpv(self):
        inputs=[4000, 1]
        settledate = datetime(2009, 1, 27)
        holding=1
        parms=[1, "USD"]
        result = DistFn_df_python_function.futpv(inputs, settledate, holding, parms)
        expected_result = 4000000000
        self.assertAlmostEqual(result, expected_result, places=0)  
        
    def test_dv01posn(self):
        security='5yr UST'
        settledate = datetime(2009, 1, 27)
        holding = 1
        riskfactor = 'USDYld5yr'
        result = DistFn_df_python_function.dv01posn(security, settledate, riskfactor, holding)
        expected_result = 481.466
        self.assertAlmostEqual(result, expected_result, places=3)
       
    def test_pvposn(self):
        security='5yr UST'
        settledate = datetime(2009, 1, 27)
        holding = 1
        marketDataValues = SamplePort_df_python_define.df_market
        result = DistFn_df_python_function.pvposn(security, holding, settledate,marketDataValues)
        expected_result = 1.097761213286204*(10**6)
        self.assertAlmostEqual(result, expected_result, places=3)     

            
    def test_pvPort(self):
        seclist = ['5yr UST','10yr UST']
        settledate = datetime(2009, 1, 27)
        holdings=[1,2]
        marketDataValues= SamplePort_df_python_define.df_market
        result = DistFn_df_python_function.pvPort(seclist, holdings, settledate,marketDataValues)
        expected_result = [1.097761213286204*(10**6), 2.225646451109351*(10**6)]
        for i in range(len(result)):
            self.assertAlmostEqual(result[i], expected_result[i], places=3)      
    def test_dv01sec(self):
        security='5yr UST'
        settledate = datetime(2009, 1, 27)
        riskfactor = 'USDYld5yr'
        result = DistFn_df_python_function.dv01sec(security, settledate, riskfactor)
        expected_result = 4.81466
        self.assertAlmostEqual(result, expected_result, places=3)
   
    def test_populateSecDV01s(self):
            seclist = ['5yr UST', '10yr UST']
            settledate = datetime(2009, 1, 27)
            holdings = [1, 2]
            df_sec = SamplePort_df_python_define.df_sec
            df_market = SamplePort_df_python_define.df_market
            df_rf = SamplePort_df_python_define.df_rf
            result = DistFn_df_python_function.populateSecDV01s(seclist, holdings, settledate, df_sec, df_rf, df_market)
            expected_result = [["5yr UST", "10yr UST"], [["USDYld5yr"], ["USDYld10yr"]], [[481.466], [1829.19]]]

            for i in range(len(result)):
                for j in range(len(result[i])):
                    if isinstance(result[i][j], list) and isinstance(expected_result[i][j], list):
                        self.assertListAlmostEqual(result[i][j], expected_result[i][j], places=2)
                    else:
                        self.assertAlmostEqual(result[i][j], expected_result[i][j], places=2)

    def assertListAlmostEqual(self, list1, list2, places=7):         ## This is a derivative function for testing test_populateSecDV01s(self
            self.assertEqual(len(list1), len(list2))
            for x, y in zip(list1, list2):
                self.assertAlmostEqual(x, y, places=places)

    def test_populatePortDV01s(self):
            seclist = ['5yr UST', '10yr UST']
            settledate = datetime(2009, 1, 27)
            holdings = [1, 2]
            df_sec = SamplePort_df_python_define.df_sec
            df_market = SamplePort_df_python_define.df_market
            df_rf = SamplePort_df_python_define.df_rf
            result = DistFn_df_python_function.populatePortDV01s(seclist, holdings, settledate,df_sec, df_rf, df_market)
            expected_result = [['USDYld5yr', 'USDYld10yr'], [481.4660345016979, 1829.1947239797096]]
            for i in range(len(result)):
                self.assertAlmostEqual(result[i], expected_result[i], places=3) 
    def test_populateEqvDV01s(self):
            seclist = ['5yr UST', '10yr UST']
            settledate = datetime(2009, 1, 27)
            df_sec = SamplePort_df_python_define.df_sec
            df_market = SamplePort_df_python_define.df_market
            df_rf = SamplePort_df_python_define.df_rf
            result = DistFn_df_python_function.populateEqvDV01s(seclist, settledate,df_sec,df_rf,df_market)
            expected_result = [['USDYld5yr', 'USDYld10yr'], [481.466, 914.597]]
            for i in range(len(result)):
              for j in range(len(result[i])):
               self.assertAlmostEqual(result[i][j], expected_result[i][j], places=2) 

if __name__ == '__main__':
            unittest.main() 
#%% 
class TestVolVCVVaRfunctions(unittest.TestCase):
    def populatevcv(self):
        rflist=["USDYld5yr", "USDYld10yr"]
        expected_result = [[50.0341, 45.2252], [45.2252, 51.0939]]
        self.assertEqual(DistFn_df_python_function.populatevcv(rflist), expected_result)
    def secVolbyVCV(self):
        seclist=['5yr UST', '10yr UST']
        holdings= [1, 2]
        settledate=datetime(2009, 1, 27)
        df_sec= SamplePort_df_python_define.df_sec
        df_rf= SamplePort_df_python_define.df_rf
        df_market= SamplePort_df_python_define.df_market
        expected_result = 16193.1
        self.assertEqual(DistFn_df_python_function.secVolbyVCV(seclist, holdings, settledate,df_sec, df_rf, df_market), expected_result) 
    def rfVolbyVCV(self):
        seclist=['5yr UST', '10yr UST']
        holdings= [1, 2]
        settledate=datetime(2009, 1, 27)
        df_sec= SamplePort_df_python_define.df_sec
        df_rf= SamplePort_df_python_define.df_rf
        df_market= SamplePort_df_python_define.df_market
        expected_result = [["USDYld5yr", "USDYld10yr"], [3405.64, 13075.1]]
        self.assertEqual(DistFn_df_python_function.rfVolbyVCV(seclist, holdings, settledate, df_sec, df_rf, df_market), expected_result) 
    def portVolbyVCV(self):
        seclist=['5yr UST', '10yr UST']
        holdings= [1, 2]
        settledate=datetime(2009, 1, 27)
        df_sec= SamplePort_df_python_define.df_sec
        df_rf= SamplePort_df_python_define.df_rf
        df_market= SamplePort_df_python_define.df_market
        expected_result = 16193.1
        self.assertEqual(DistFn_df_python_function.portVolbyVCV(seclist, holdings, settledate, df_sec, df_rf, df_market), expected_result) 
    def portVolbyVCVextended(self):
        seclist=['5yr UST', '10yr UST']
        holdings= [1, 2]
        settledate=datetime(2009, 1, 27)
        df_sec= SamplePort_df_python_define.df_sec
        df_rf= SamplePort_df_python_define.df_rf
        df_market= SamplePort_df_python_define.df_market
        expected_result = [16193.1, [3175.93, 13017.1], [0.196129, 0.803871], [3117.98, 
  12787.4], [-3.43406, -0.465958], [5846.4, 1522.8], [3405.64, 
  13075.1], ["USDYld5yr", "USDYld10yr"], [481.466, 
  914.597], [0.932549, 0.995568], [4.43406, 2.46596], [481.466, 
  1829.19], [[50.0341, 45.2252], [45.2252, 51.0939]]]
        self.assertEqual(DistFn_df_python_function.portVolbyVCVextended(seclist, holdings, settledate, df_sec, df_rf, df_market), expected_result)

if __name__ == '__main__':
    unittest.main()