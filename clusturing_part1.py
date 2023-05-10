# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:14:54 2023

@author: vasth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
# from sklearn import cluster
import sklearn.cluster as cluster
import sklearn.metrics as skmet

# Defining a function read_world_bank_csv
def read_world_bank_csv(file_name):
    
    """
    This function reads a CSV file from the World Bank data
    and returns two DataFrames.

    Parameters
    ----------
    file_name : str
    The path to the CSV file.

    Returns
    -------
    year_as_column : DataFrame
    The DataFrame containing the data from the CSV file with years as columns.

    country_as_column : DataFrame
    The DataFrame containing the data from the CSV file, transposed with 
    countries as columns.

    Raises
    ------
    FileNotFoundError
    If the CSV file does not exist.

    IOError
    If there is an error reading the CSV file.

    """
    
    # Reading the CSV file into a DataFrame
    year_as_column = pd.read_csv(file_name, header = 2)

    # Dropping the first three columns, which are the header, 
    # Indicator Code, and Country Code.
    year_as_column = year_as_column.drop(['Indicator Code', 'Country Code',
                                          'Indicator Name'], axis = 1)

    # Dropping any rows that contain missing values and cleaning dataframe.
    year_as_column.dropna()
    year_as_column.dropna(how = 'all', axis = 1, inplace = True)
    
    # Setting the index of the DataFrame to the Country Name column.
    year_as_column = year_as_column.set_index('Country Name')

    # Renaming the axis of the DataFrame to Years.
    year_as_column = year_as_column.rename_axis(index = 'Country Name',
                                                columns = 'Year')

    # Transposing the DataFrame.
    country_as_column = year_as_column.transpose()

    # Renaming the axis of the DataFrame to Countries.
    country_as_column = country_as_column.rename_axis(columns = 'Country Name', 
                                                      index = 'Year')

    # Dropping the first row of the DataFrame, which is just the column names.
    country_as_column = country_as_column.iloc[1:]

    # Returns the two DataFrames.
    return year_as_column, country_as_column



df_ate, df_atet = read_world_bank_csv('Access to Electricity(% of Population).csv')
print(df_ate.head())

df_co3 = df_ate[["1990", "2000", "2010", "2020"]]
print(df_co3.head())


pd.plotting.scatter_matrix(df_co3, figsize=(12, 12), s=5, alpha=0.8)
plt.show()

df_ex = df_co3[["1990", "2020"]] # extract the two columns for clustering
df_ex = df_ex.dropna() # entries with one nan are useless
df_ex = df_ex.reset_index()
print(df_ex.iloc[0:15])
# reset_index() moved the old index into column index
# remove before clustering
df_ex = df_ex.drop("Country Name", axis=1)
print(df_ex.iloc[0:15])
# normalise, store minimum and maximum
df_norm, df_min, df_max = ct.scaler(df_ex)
print()
print("n score")
# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm) # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_ex, labels))
    

ncluster = 7 # best number of clusters
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["1990"], df_norm["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("ATE(1990)")
plt.ylabel("ATE(2020)")
plt.show()

print(cen)
# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)
xcen = scen[:, 0]
ycen = scen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_ex["1990"], df_ex["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("ATE(1990)")
plt.ylabel("ATE(2020)")
plt.show()


df_agrar, methane_t = read_world_bank_csv(
    'Methane emissions (kt of CO2 equivalent).csv')
df_forest, urban_pop_t = read_world_bank_csv('Urban population.csv')
print(df_agrar.describe())
print(df_forest.describe())

# drop rows with nan's in 2019
df_agrar = df_agrar[df_agrar["2019"].notna()]
print(df_agrar.describe())
# alternative way of targetting one or more columns
df_forest = df_forest.dropna(subset=["2019"])
print(df_forest.describe)

df_agr2020 = df_agrar[["2019"]].copy()
df_for2020 = df_forest[[ "2019"]].copy()

print(df_agr2020.describe())
print(df_for2020.describe())

df_2020 = pd.merge(df_agr2020, df_for2020, on="Country Name", how="outer")
print(df_2020.describe())
df_2020.to_excel("agr_for2020.xlsx")

print(df_2020.describe())
df_2020 = df_2020.dropna() # entries with one datum or less are useless.
print()
print(df_2020.describe())
# rename columns
df_2020 = df_2020.rename(columns={"2019_x":"Methane emissions", "2019_y":"Urban population"})
pd.plotting.scatter_matrix(df_2020, figsize=(12, 12), s=5, alpha=0.8)
print(df_2020.corr())

df_cluster = df_2020[["Methane emissions", "Urban population"]].copy()
# normalise
df_cluster, df_min, df_max = ct.scaler(df_cluster)

print("n score")
# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_cluster) # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_cluster, labels))
    

ncluster = 5
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_cluster) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_cluster["Methane emissions"], df_cluster["Urban population"], 10, labels,marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Methane emissions")
plt.ylabel("Urban population")
plt.show()

# move the cluster centres to the original scale
cen = ct.backscale(cen, df_min, df_max)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_2020["Methane emissions"], df_2020["Urban population"], 10, labels, marker="o",cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Methane emissions")
plt.ylabel("Urban population")
plt.show()