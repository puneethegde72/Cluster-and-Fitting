# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:46:00 2023

@author: Puneet
"""
'''Importing all Required libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.optimize as opt

"""Reading manipulating file with country name
and returning a dataframe and transpose of the dataframe as return"""
def dataFrame(file_name, col, value1,countries):
    # Reading Data for dataframe
    df = pd.read_csv(file_name, skiprows = 4)
    # Grouping data with col value
    df1 = df.groupby(col, group_keys = True)
    #retriving the data with the all the group element
    df1 = df1.get_group(value1)
    #Reseting the index of the dataframe
    df1 = df1.reset_index()
    #Storing the column data in a variable
    a = df1['Country Name']
    # cropping the data from dataframe
    df1 = df1.iloc[countries,3:]
    df1 = df1.drop(columns=['Indicator Name', 'Indicator Code'])
    df1.insert(loc=0, column='Country Name', value=a)
    #Dropping the NAN values from dataframe Column wise
    df1= df1.dropna(axis = 1)
    #transposing the index of the dataframe
    df2 = df1.set_index('Country Name').T
    #returning the normal dataframe and transposed dataframe
    return df1, df2


# countries which are using for data analysis
countries = [35, 55, 81, 109]
'''calling dataFrame functions for all the dataframe which will be used for K-Means'''
ele_con_c, ele_con_y = dataFrame("API_19_DS2_en_csv_v2_4700503.csv",
                                       "Indicator Name", "Electric power consumption (kWh per capita)",
                                       countries)
#Printing value by countries
print(ele_con_c,)

#Printing value by Year
print(ele_con_y)

#returns a numpy array as x
x = ele_con_y.values

'''Function to normalize the dataframe'''
def normalizing(value):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    return data

normalized_df = normalizing(x)
print(normalized_df)
'''function to find the no of cluster needed by elbow method'''
def n_cluster(dataFrame,n):
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    return wcss

k = n_cluster(normalized_df,10)
print(k)
'''Visualization of Elbow method'''
plt.figure()
plt.plot(range(1, 10), k)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

#finding k means cluster
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)
#fitting and predicting the data using k means cluster

lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids= kmeans.cluster_centers_
print('centroids=',centroids)

'''Ploting Kmeans clusters'''
plt.figure()
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(normalized_df.values[lables == 2, 0], normalized_df.values[lables == 2, 1], s = 100, c = '#EE3A8C', label = 'Cluster3')
#Ploting cluster 4
plt.scatter(normalized_df.values[lables == 3, 0], normalized_df.values[lables == 3, 1], s = 100, c = '#43CD80', label = 'Cluster4')
#Ploting cluster 5
plt.scatter(normalized_df.values[lables == 4, 0], normalized_df.values[lables == 4, 1], s = 100, c = '#8EE5EE', label = 'Cluster5')
#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')

plt.legend()
# Title of the  plot
plt.title('Clusters of GDP per Capita of 4 countries for year 1971 to 2014')
plt.xlabel('Years')
plt.ylabel('GDP per year')
plt.show()


'''calling dataFrame functions for all the dataframe which will be used for curve fitting'''
school_c, school_y = dataFrame("API_19_DS2_en_csv_v2_4700503.csv",
                                       "Indicator Name", "School enrollment, primary and secondary (gross), gender parity index (GPI)",countries)
school_y['years'] = school_y.index

#Ploting the data of the indian school enrolment for primar and secondary
school_y.plot(y='India',use_index=True)

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1999.0
    f = n0 * np.exp(g*t)
    return f

print(type(school_y["years"].iloc[1]))
school_y["years"] = pd.to_numeric(school_y["years"])
print(type(school_y["years"].iloc[1]))
param, covar = opt.curve_fit(exponential, school_y["years"], school_y["India"],
p0=(73233967692.102798, 0.10))

school_y["fit"] = exponential(school_y["years"], *param)
school_y.plot("years", ["India", "fit"])
plt.show()

print(school_y)

'''function for logistic fit which will be used for prediction'''
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

param, covar = opt.curve_fit(logistic, school_y["years"], school_y["India"],
                             p0=(3e12, 0.03, 2000.0), maxfev=5000)

sigma = np.sqrt(np.diag(covar))
igma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
school_y["logistic function fit"] = logistic(school_y["India"], *param)
school_y.plot("years", ["India", "fit"])
plt.show()

#predicting years
year = np.arange(1960, 2035)
print(year)
forecast = logistic(year, *param)

'''Predicting the values with plot'''
plt.figure()
plt.plot(school_y["years"], school_y["India"], label="School enrollment")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("Indian education")
plt.legend()
plt.show()