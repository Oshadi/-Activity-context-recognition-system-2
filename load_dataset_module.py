import os.path

import numpy as np
import pandas as pd
import traceback

#from feature_module import Feature_Module
import feature_module

#function to do the sliding window
def sliding_windows(dataset, winsize, transform):
    ranged = np.arange(len(dataset))
    steped = ranged[::transform]
    chunk = list(zip(steped, steped + winsize))
    form = '{0[0]}:{0[1]}'.format
    tdata = lambda transform: dataset.iloc[transform[0]:transform[1]]
    return pd.concat(map(tdata, chunk), keys=map(form, chunk))



class Load_Dataset_Module:

    def activity_data(file_path):
        #fileExist = os.path.exists(file_path)
        try:
            dataSet = pd.read_csv(file_path)
            x = np.array(dataSet.iloc[:, 1:19])
            df = pd.DataFrame(x, columns=('orX','orY','orZ','rX','rY','rZ','accX','accY','accZ','gX','gY','gZ','mX','mY','mZ','lux','soundLevel','activity'))
            df['orientation'] = df[df.columns[0:3]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
            )
            df['rotation'] = df[df.columns[3:6]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
            )
            df['accelerometer'] = df[df.columns[6:9]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
            )
            df['gyroscope'] = df[df.columns[9:12]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
            )
            df['magnetic_sensors'] = df[df.columns[12:15]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
            )
            dropedcol=df.drop(['orX','orY','orZ','rX','rY','rZ','accX','accY','accZ','gX','gY','gZ','mX','mY','mZ'], axis=1)
            dropedcol = dropedcol[["orientation", "rotation","accelerometer","gyroscope","magnetic_sensors","lux","soundLevel","activity"]]
                #dropedcol.to_csv("myFormattedCSV.csv")
            slid_win=sliding_windows(dropedcol,10000,5000)
                #slid_win.to_csv("oshadi.csv")
            return slid_win

        except:
            traceback.print_exc()
            print("Ohhhhh! Something went wrong! Please try again later!")

    def activity_features(data):
        
        #separating the data
        orientationSet=(data.loc[:, "orientation"])
        rotationSet = (data.loc[:, "rotation"])
        accelerometerSet = (data.loc[:, "accelerometer"])
        gyroscopeSet = (data.loc[:, "gyroscope"])
        magnetic_sensorsSet = (data.loc[:, "magnetic_sensors"])
        
        #object creation for each sensor
        featureMoOBJ = feature_module.Feature_Module(orientationSet)
        featureMoOBJRo = feature_module.Feature_Module(rotationSet)
        featureMoOBJAcc = feature_module.Feature_Module(accelerometerSet)
        featureMoOBJGyr = feature_module.Feature_Module(gyroscopeSet)
        featureMoOBJMS = feature_module.Feature_Module(magnetic_sensorsSet)


        #accessing the feature_module functions
        row1Dict = {}
        row1Dict['orientationMean'] = featureMoOBJ.Mean()
        row1Dict['orientationMedian'] = featureMoOBJ.Median()
        row1Dict['orientationSTD'] = featureMoOBJ.Standard_deviation()
        row1Dict['orientationVariance'] = featureMoOBJ.Varience()
        row1Dict['orientationRMS'] = featureMoOBJ.Root_mean_square()
        row1Dict['orientationSOS'] = featureMoOBJ.Sum_Of_Squares()
        row1Dict['orientationCovari'] = featureMoOBJ.Covariance()
        row1Dict['orientationZCR'] = featureMoOBJ.Zero_Crossing()

        row1Dict['rotationMean'] = featureMoOBJRo.Mean()
        row1Dict['rotationMedian'] = featureMoOBJRo.Median()
        row1Dict['rotationSTD'] = featureMoOBJRo.Standard_deviation()
        row1Dict['rotationVariance'] = featureMoOBJRo.Varience()
        row1Dict['rotationRMS'] = featureMoOBJRo.Root_mean_square()
        row1Dict['rotationSOS'] = featureMoOBJRo.Sum_Of_Squares()
        row1Dict['rotationCovari'] = featureMoOBJRo.Covariance()
        row1Dict['rotationZCR'] = featureMoOBJRo.Zero_Crossing()

        row1Dict['accelerometerMean'] = featureMoOBJAcc.Mean()
        row1Dict['accelerometerMedian'] = featureMoOBJAcc.Median()
        row1Dict['accelerometerSTD'] = featureMoOBJAcc.Standard_deviation()
        row1Dict['accelerometerVariance'] = featureMoOBJAcc.Varience()
        row1Dict['accelerometerRMS'] = featureMoOBJAcc.Root_mean_square()
        row1Dict['accelerometerSOS'] = featureMoOBJAcc.Sum_Of_Squares()
        row1Dict['accelerometerCovari'] = featureMoOBJAcc.Covariance()
        row1Dict['accelerometerZCR'] = featureMoOBJAcc.Zero_Crossing()

        row1Dict['gyroscopeMean'] = featureMoOBJGyr.Mean()
        row1Dict['gyroscopeMedian'] = featureMoOBJGyr.Median()
        row1Dict['gyroscopeSTD'] = featureMoOBJGyr.Standard_deviation()
        row1Dict['gyroscopeVariance'] = featureMoOBJGyr.Varience()
        row1Dict['gyroscopeRMS'] = featureMoOBJGyr.Root_mean_square()
        row1Dict['gyroscopeSOS'] = featureMoOBJGyr.Sum_Of_Squares()
        row1Dict['gyroscopeCovari'] = featureMoOBJGyr.Covariance()
        row1Dict['gyroscopeZCR'] = featureMoOBJGyr.Zero_Crossing()

        row1Dict['magnetic_sensorsMean'] = featureMoOBJMS.Mean()
        row1Dict['magnetic_sensorsMedian'] = featureMoOBJMS.Median()
        row1Dict['magnetic_sensorsSTD'] = featureMoOBJMS.Standard_deviation()
        row1Dict['magnetic_sensorsVariance'] = featureMoOBJMS.Varience()
        row1Dict['magnetic_sensorsRMS'] = featureMoOBJMS.Root_mean_square()
        row1Dict['magnetic_sensorsSOS'] = featureMoOBJMS.Sum_Of_Squares()
        row1Dict['magnetic_sensorsCovari'] = featureMoOBJMS.Covariance()
        row1Dict['magnetic_sensorsZCR'] = featureMoOBJMS.Zero_Crossing()

        row1Dict['lux'] = (data.loc[:, "lux"])
        row1Dict['soundLevel'] = (data.loc[:, "soundLevel"])
        row1Dict['activity'] = (data.loc[:, "activity"])

        #conversion for a dataframe
        featured_dataSet=pd.DataFrame.from_dict(row1Dict)
        return featured_dataSet
    
        #tocsv.to_csv("myFullSet3.csv")

#     data_obj=activity_data("/Users/oshadirathnayake/Documents/PCP/Pycharm/activity_context_tracking_data.csv")
#     feature_module_OBJ = activity_features(data_obj)

