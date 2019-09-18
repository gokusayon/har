# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:51:46 2019

@author: vasum
"""
# importing libraries and dependecies 
import pandas as pd
import numpy as np
from math import sqrt, atan2, radians


# define integrator class for use in comp filter(rectangular for now)
class Integrator:
    def __init__(self, SamplePeriod = 1., InitialCondition = 0.):
        self.T = SamplePeriod
        self.State = InitialCondition
    def __call__(self, Input=0.):
        self.State += self.T * Input
        return self.State
        
# define comp filter class
class ComplementaryFilter:
    def __init__(self,SamplePeriod,BandWidth,Gyro_zero,On_Axis_zero,Off_Axis_Zero,Z_Axis_zero,One_Gee):
        self.k = BandWidth
        self.one_gee = One_Gee
        self.Internal = Integrator(SamplePeriod,-Gyro_zero)
        self.Perpendicular = sqrt(Off_Axis_Zero**2 + Z_Axis_zero**2)
        self.Output = Integrator(SamplePeriod,-atan2(On_Axis_zero,self.Perpendicular))
        self.Prev_Output = self.Output()
        
    def __call__(self,Gyro_input,On_Axis_input,Off_Axis_input,Z_Axis_input):
        self.gmag = sqrt(On_Axis_input**2 + Off_Axis_input**2 + Z_Axis_input**2)
        self.Gyro_in = Gyro_input
        self.Perpendicular = sqrt(Off_Axis_input**2 + Z_Axis_input**2)
        self.angle = -atan2(On_Axis_input,self.Perpendicular)
        if  abs(self.gmag-self.one_gee)/self.one_gee >0.05:
            self.input1 = 0.
        else:
            self.input1 = (self.angle - self.Prev_Output)
        self.temp = self.Internal(self.input1*self.k*self.k)    
        self.input2 = self.temp + (self.input1)*2*self.k - Gyro_input
        self.temp = self.Output(self.input2)
        self.Prev_Output = self.temp
        return self.temp

# define comp filter class
class Filter:
    
    combined_dataset = 0
    acc = 0
    gyro = 0
    maxValue = 0
    isEnable = False
    
        
        
    
    def getProcessedData(self, forceEnable = 0):
        
        len1 = self.acc.shape[0]
        len2 = self.acc.shape[0]
        
        if(len1 != len2):
            self.isEnable = False
            print("Size of two dataset dow not match")
        else:
            self.isEnable = True
            self.maxValue = len1
            
        print("Processing data : ")
        k = 4. #was 0.25
        T = 1/100.
        AREF = 2.82
        
        ACCEL_ZERO_G = 10
        
        time ,xaccavg ,yaccavg ,zaccavg = np.mean(self.acc,axis=0)
        time ,xgyrozero ,ygyrozero ,zgyrozero = np.mean(self.gyro,axis=0)
        x_com =0;
        y_comp =0;
        for i in range(self.maxValue):
           # =====================Intitalization only===================
# =============================================================================
#             time,xaccel,yaccel,zaccel = self.acc.iloc[i]
#             time2,xgyro,ygyro,zgyro = self.gyro.iloc[i]
#             
#             xgyrodeg = (xgyro - xgyrozero)/1.024 * AREF / 2. 
#             ygyrodeg = (ygyro - ygyrozero)/1.024 * AREF / 2. 
#             zgyrodeg = (zgyro - zgyrozero)/1.024 * AREF / 3.3
#             xacc = xaccel - ACCEL_ZERO_G
#             yacc = yaccel - ACCEL_ZERO_G
#             zacc = zaccel - ACCEL_ZERO_G
#                 
#             
#             one_gee = sqrt(xacc**2 + yacc**2 + zacc**2)
#             
#             
#                 
#             anglez = 0.
#             
#             my_x_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
#                                 Gyro_zero = 0., On_Axis_zero = yacc,Off_Axis_Zero=xacc,
#                                 Z_Axis_zero=zacc,One_Gee=one_gee)
#              
#             my_y_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
#                                 Gyro_zero = 0., On_Axis_zero = xacc,Off_Axis_Zero=zacc,
#                                 Z_Axis_zero=zacc,One_Gee=one_gee)
#             
# =============================================================================
            # get intial dataset to initialize the gyro-only cube
            dtx = -radians(xgyrodeg * T)
            dty = radians(ygyrodeg * T)
            dtz = -radians(zgyrodeg * T)
            x_comp_filter = my_x_comp_filter(Gyro_input=radians(xgyrodeg),
                                On_Axis_input=yacc, Off_Axis_input=xacc,Z_Axis_input=zacc)
            
            y_comp_filter = my_y_comp_filter(Gyro_input=radians(ygyrodeg),
                                On_Axis_input=xacc, Off_Axis_input=yacc,Z_Axis_input=zacc)
            
            
            # There is no comp filter in yaw
            # Put a deadzone of about 1.5 LSB to supress drift (may not work in-flight)
            if abs(radians(zgyrodeg)) > 0.03:
                anglez += dtz
                
        #    print(time," : ",-radians(xgyrodeg), " : " , -radians(ygyrodeg)," : " , radians(zgyrodeg)," : " , x_comp_filter," : " , 
        #          y_comp_filter," : " , xacc," : " , yacc," : " , -zacc," : " , anglez)
            
            self.combined_dataset.loc[i] = [time ,-radians(xgyrodeg),-radians(ygyrodeg),radians(anglez),x_comp_filter,y_comp_filter,xacc,yacc,-zacc,anglez,dtx,dty]
        
            
#            if(i%1000 == 999):
#                print(".")
                
        
        return self.combined_dataset
    
    def __init__(self,acc_data_path,gyro_data_path):
        
#        acc = pd.read_csv("Dataset/walking/accelerometer.csv")
#        gyro = pd.read_csv("Dataset/walking/gyroscope.csv")
        self.acc =0
        self.gyro =0
        self.combined_dataset = pd.DataFrame(columns=["time" ,"xgyrodeg","ygyrodeg","zgyrodeg","x_comp_filter","y_comp_filter","xacc"
                                                 ,"yacc","zacc","anglez","angley","anglex"])
        self.acc = pd.read_csv(acc_data_path)
        self.gyro = pd.read_csv(gyro_data_path)


    