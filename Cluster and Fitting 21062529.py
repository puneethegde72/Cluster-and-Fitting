# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:46:00 2023

@author: Puneet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sts
from sklearn.cluster import KMeans

"""Reading manipulating file with country name
and returning a dataframe and transpose of the dataframe as return"""
def dataFrame(file_name, years, countries, col, value1):
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
    df1 = df1.iloc[countries, years:]
    df1.insert(loc=0, column='Country Name', value=a)
    #Dropping the NAN values from dataframe Column wise
    df1= df1.dropna(axis = 1)
    #transposing the index of the dataframe
    df2 = df1.set_index('Country Name').T
    #returning the normal dataframe and transposed dataframe
    return df1, df2
# years using for the data analysis
years = 35
# countries which are using for data analysis
countries = [35, 40, 55, 81, 109, 119, 202, 205, 251]

'''calling dataFrame functions for all the dataframe which will be used for visualization'''
GDP_capita_c, GDP_capita_y = dataFrame("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv", years,
                                     countries, "Indicator Name", "GDP per capita (current US$)")
print(GDP_capita_c)
print(GDP_capita_y)

wcss = []
df1=GDP_capita_c.iloc[:,1:]
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 0)
    kmeans.fit(df1)
    wcss.append(kmeans.inertia_)
print(wcss)