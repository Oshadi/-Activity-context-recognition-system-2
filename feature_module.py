import csv
import numpy as np
import pandas as pd

class Feature_Module:
    def __init__(self, dataSet):
        self.dataSet = dataSet


    # function to calculate Mean
    def Mean(self):
        mean_array=[]
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            mean=np.mean(arr)
            mean_array.append(mean)
        return mean_array


    # calculate Standard_deviation
    def Standard_deviation(self):
        std_array = []
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            std=np.std(arr)
            std_array.append(std)
        return std_array
        #stDev = np.std(self.dataSet)


    # calculate Varience
    def Varience(self):
        variance_array = []
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            median = np.var(arr)
            variance_array.append(median)
        #vari = np.var(self.dataSet)
        return variance_array


    # calculate Median
    def Median(self):
        median_array = []
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            median = np.median(arr)
            median_array.append(median)
        #med=self.dataSet.loc[:].median()
        return median_array



    # calculate Root mean square
    def Root_mean_square(self):
        rms_array = []
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            rms = np.sqrt(np.mean(arr ** 2))
            rms_array.append(rms)
        return rms_array


    # calculate Sum of squares
    def Sum_Of_Squares(self):
        sos_array = []
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            sos = np.sqrt(np.mean(arr ** 2))
            sos_array.append(sos)
        return sos_array


    # calculate Zero Crossing
    def Zero_Crossing(self):
        zcr_array = []
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            zcr = np.nonzero(np.diff(arr > 0))[0]
            zeCros = zcr.size
            zcr_array.append(zeCros)
        return zcr_array


    # calculate Covariance
    def Covariance(self):
        cov_array = []
        for item in self.dataSet:
            my_input = []
            x = item.split(',')
            for i in x:
                p = float(i)
                my_input.append(p)
                arr = np.array(my_input)
            cov = np.cov(arr)
            cov_array.append(cov)
        return cov_array






