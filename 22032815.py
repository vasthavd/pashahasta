# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:46:39 2023

@author: vasth
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
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



# Defining a function correlation_heatmap
def correlation_heatmap(year):
    
    """
    This function creates a correlation heatmap for a given year by merging 
    data from various sources and calculating a correlation matrix 
    between the variables.

    Parameters
    ----------
    year : int 
    The year for which the correlation heatmap needs to be created.

    Returns
    -------
    None , Plots a correlation heatmap.
    
    """
    
    # Merge all the dataframes required for the given year
    m_df = pd.merge(ate[year], urban_pop[year],
                    left_index = True, right_index = True)
    m_df = pd.merge(m_df, methane[year], left_index = True, right_index = True)
    m_df = pd.merge(m_df, forest_area[year],
                    left_index = True, right_index = True)
    m_df = pd.merge(m_df, gdp[year], left_index = True, right_index = True)
    m_df = pd.merge(m_df, agri[year], left_index = True, right_index = True)
    m_df = pd.merge(m_df, ce[year], left_index = True, right_index = True)
    m_df = pd.merge(m_df, gpi[year], left_index = True, right_index = True)
    m_df.columns = ['Electricity', 'Urban_pop', 'Methane', 'Forest',
                  'GDP', 'Agri land', 'C02', 'GPI']

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Calculate the correlation matrix
    corr_matrix = m_df.corr()

    # Create a heatmap
    heat_map = ax.imshow(corr_matrix, cmap = 'coolwarm')

    # Set the ticks and tick labels
    ax.set_xticks(range(len(m_df.columns)))
    ax.set_yticks(range(len(m_df.columns)))
    ax.set_xticklabels(m_df.columns)
    ax.set_yticklabels(m_df.columns)

    # Rotate the tick labels and set them at the center
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right',
             rotation_mode = 'anchor')

    # Add the correlation values inside the heatmap
    for i in range(len(m_df.columns)):
        for j in range(len(m_df.columns)):
            text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2),
                           ha = "center", va = "center", 
                           color = "w", rotation = 45)

    # Add a colorbar
    cbar = ax.figure.colorbar(heat_map, ax = ax)

    # Set the title and show the plot
    ax.set_title(f"Correlation Heatmap for {year}")
    plt.tight_layout()




def form_cluster(read_data, label):
    
    """
    This function reads data from the given dataframe 
    and selects only the columns with years 1990, 2000, 2010, and 2020. 
    It then generates a scatter matrix plot of the data 
    and extracts the two columns (specified by xlabel and ylabel) 
    for clustering. The extracted data is then normalized and 
    clustered using K-means clustering algorithm for different 
    values of k (number of clusters), and the silhouette score is 
    calculated for each k value. Based on the silhouette scores, 
    the function selects the best number of clusters and generates 
    two cluster plots: one on the normalized scale and another 
    on the original scale. The function also displays the estimated 
    cluster centers for each selected k value.
    

    Parameters
    ----------
    - read_data (pandas.DataFrame): Input dataframe.
    - xlabel (str): Label for the x-axis in the cluster plot.
    - ylabel (str): Label for the y-axis in the cluster plot.

    Returns
    -------
    - None: The function does not return any value, 
    but it generates and displays cluster plots.

    Raises
    ------
    - No specific exceptions are raised by this function.

    """

    # Select only the columns with years 1990, 1999, 2009, and 2019.
    read_data_year = read_data[["1990", "1999", "2009", "2019"]]

    # Generate a scatter matrix plot of the data.
    fig, ax = plt.subplots(figsize=(12, 12))
    pd.plotting.scatter_matrix(read_data_year, ax=ax, s=5, alpha=0.8)

    # Add a title to the plot
    plt.suptitle("Scatter Matrix Plot of {}".format(label), y=0.95, fontsize=16)

    # Display the plot
    plt.show()

    # Extract the two columns (specified by xlabel and ylabel) for clustering.
    form_cluster = read_data_year[["1990", "2019"]]

    # Remove any rows with missing values.
    form_cluster = form_cluster.dropna()

    # Reset the index of the DataFrame.
    form_cluster = form_cluster.reset_index()

    # Drop the "Country Name" column from the DataFrame.
    form_cluster = form_cluster.drop("Country Name", axis=1)

    # Print the first 15 rows of the DataFrame.
    print(form_cluster.iloc[0:15])

    # Normalize the data.
    df_norm, df_min, df_max = ct.scaler(form_cluster)

    # Print the silhouette scores for different values of k.
    for ncluster in range(2, 10):

        # Create a K-means clustering model with k clusters.
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Fit the model to the data.
        kmeans.fit(df_norm)

        # Extract the labels for each data point.
        labels = kmeans.labels_

        # Calculate the silhouette score for the model.
        silhoutte_score = skmet.silhouette_score(form_cluster, labels)

        # Print the silhouette score.
        print(ncluster, silhoutte_score)

    # Select the number of clusters with the best silhouette score.
    ncluster = 7

    # Create a K-means clustering model with k clusters.
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the model to the data.
    kmeans.fit(df_norm)

    # Extract the labels for each data point.
    labels = kmeans.labels_

    # Extract the estimated cluster centers.
    cen = kmeans.cluster_centers_

    # Convert the cluster centers to NumPy arrays.
    cen = np.array(cen)

    # Extract the x-coordinates of the cluster centers.
    xcen = cen[:, 0]

    # Extract the y-coordinates of the cluster centers.
    ycen = cen[:, 1]

    # Set the figure size
    plt.figure(figsize=(8.0, 8.0))

    # Get the color map
    cm = plt.cm.get_cmap('tab10')

    # Scatter plot the normalized data points colored by the cluster with specified markersize, marker and cmap
    plt.scatter(df_norm["1990"], df_norm["2019"], 10, labels, marker="o", cmap=cm)

    # Scatter plot the cluster centroids with specified markersize, marker and color
    plt.scatter(xcen, ycen, 45, "k", marker="d")

    # Set the label for x-axis
    plt.xlabel("{} (1990)".format(label))

    # Set the label for y-axis
    plt.ylabel("{} (2019)".format(label))

    # Set the title with dynamic label values
    plt.title("Cluster plot of {}".format(label))

    # Show the plot
    plt.show()

    # Print the cluster centroids
    print(cen)

    # Backscale the normalized cluster centers to original scale
    scen = ct.backscale(cen, df_min, df_max)

    # Print the back-scaled cluster centers
    print()
    print(scen)

    # Get the x-coordinate of cluster centers
    xcen = scen[:, 0]

    # Get the y-coordinate of cluster centers
    ycen = scen[:, 1]

    # Set the figure size
    plt.figure(figsize=(8.0, 8.0))

    # Get the color map
    cm = plt.cm.get_cmap('tab10')

    # Scatter plot the data points colored by the cluster with specified markersize, marker and cmap
    plt.scatter(form_cluster["1990"], form_cluster["2019"], 10, labels, marker="o", cmap=cm)

    # Scatter plot the cluster centroids with specified markersize, marker and color
    plt.scatter(xcen, ycen, 45, "k", marker="d")

    # Set the label for x-axis
    plt.xlabel("{} (1990)".format(label))

    # Set the label for y-axis
    plt.ylabel("{} (2019)".format(label))

    # Set the title with dynamic label values
    plt.title("Cluster plot of {} on original scale".format(label))

    # Show the plot
    plt.show()




def analyze_clusters(analyze_df, analyze_dfb, xlabel, ylabel):
    
    """
    Analyzes the clusters of data in two input pandas DataFrames.

    Parameters
    ----------
       - analyze_df (pandas.DataFrame): The first dataframe.
       - analyze_dfb (pandas.DataFrame):The second dataframe.
       - xlabel (str): The name of the x-axis column in the dataframe.
       - ylabel (str): The name of the y-axis column in the dataframe.

    Returns
    -------
        - None.

    """

    # print descriptive statistics for analyze_df and analyze_dfb
    print(analyze_df.describe())
    print(analyze_dfb.describe())

    # drop rows with missing values in the "2019" column for analyze_df and analyze_dfb
    analyze_df = analyze_df[analyze_df["2019"].notna()]
    print(analyze_df.describe())

    # an alternative way of dropping rows with missing values in the "2019" column for analyze_dfb
    analyze_dfb = analyze_dfb.dropna(subset=["2019"])
    print(analyze_dfb.describe)

    # extract the "2019" column from analyze_df and analyze_dfb
    year_extracted = analyze_df[["2019"]].copy()
    year_extractedb = analyze_dfb[[ "2019"]].copy()

    # print descriptive statistics for year_extracted and year_extractedb
    print(year_extracted.describe())
    print(year_extractedb.describe())

    # merge year_extracted and year_extractedb on the "Country Name" column
    # and get a new data frame with the merged "2019" column
    year_df = pd.merge(year_extracted, year_extractedb, on="Country Name", how="outer")

    # print descriptive statistics for year_df
    print(year_df.describe())

    # save year_df to an Excel file
    year_df.to_excel("agr_for2020.xlsx")

    # drop rows with missing values in year_df
    # entries with one datum or less are useless.
    year_df = year_df.dropna()

    # print descriptive statistics for year_df
    print(year_df.describe())

    # rename the "2019_x" and "2019_y" columns of year_df
    # to the xlabel and ylabel variables, respectively
    year_df = year_df.rename(columns={"2019_x":xlabel, "2019_y":ylabel})

    # plot a scatter matrix of year_df with a figure size of (12, 12), point size of 5, and alpha of 0.8
    fig, ax = plt.subplots(figsize=(12, 12))
    pd.plotting.scatter_matrix(year_df, ax=ax, s=5, alpha=0.8)

    # add a title to the plot
    plt.suptitle("Scatter Matrix of 2019 Data", y=0.95, fontsize=16)

    # display the plot
    plt.show()

    # calculate the correlation of xlabel and ylabel columns of year_df
    # and print the resulting correlation matrix
    print(year_df.corr())

    # copy the xlabel and ylabel columns of year_df to df_cluster
    df_cluster = year_df[[xlabel, ylabel]].copy()

    # normalize the data in df_cluster using the scaler() function
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
        

    # Set the number of expected clusters
    ncluster = 5

    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_cluster)  # Fit done on x,y pairs

    # Extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # Cluster by cluster plot
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_cluster[xlabel], df_cluster[ylabel], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set the title with dynamic label values
    plt.title("Cluster analysis of 2019")
    plt.show()

    # Move the cluster centres to the original scale
    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # Cluster by cluster plot on original scale
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(year_df[xlabel], year_df[ylabel], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set the title with dynamic label values
    plt.title("Cluster analysis of 2019 original scale")
    plt.show()



def get_country_data(df, country, value):
    
    """
    Returns a pandas DataFrame containing the data for a given country 
    and a specific value.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame containing the data.
    country : str
        The name of the country to filter the data for.
    value : str
        The name of the value to extract from the data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the year and the specified value 
        for the given country.
    """
    # select columns for given country and drop null values
    country_data = df.loc[:,[country]].reset_index().dropna()
        
    # rename columns as Year and the specified value
    country_data.columns = ['Year', value]
        
    # return the DataFrame
    return country_data




def country_data_analysis_plot(country_data, xlabel, ylabel, country):
    
    """
    Creates and displays plots of different fit functions for 
    a given country's data.

    Parameters
    ----------
        country_data (pandas.DataFrame): A DataFrame containing the 
        country's data.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        country (str): The name of the country.

    Returns:
        None
    """
    
    def exponential(t, n0, g):
        
        """
        Calculates an exponential function with scale factor n0 
        and growth rate g.
    
        Parameters
        ----------
            - t (float): The input variable of the function.
            - n0 (float): The scale factor of the exponential function.
            - g (float): The growth rate of the exponential function.
    
        Returns
        -------
            - f (float): The result of the exponential function.
        """
        
        # Mathematical calculations and returns a function value
        t = t - 1990
        f = n0 * np.exp(g*t)
        return f


    def poly(x, a, b, c, d, e):
        
        """
        Calculates a polynomial function.
    
        Parameters
        ----------
            - x (float): The input variable of the function.
            - a (float): The constant term of the polynomial.
            - b (float): The coefficient of x.
            - c (float): The coefficient of x^2.
            - d (float): The coefficient of x^3.
            - e (float): The coefficient of x^4.
    
        Returns
        -------
            - f (float): The result of the polynomial function.
        """
        
        # Mathematical calculations and returns a function value
        x = x - 1990
        f = a + b*x + c*x**2 + d*x**3 + e*x**4
        return f


    def logistic(t, n0, g, t0):
        """
        Calculates a logistic function with scale factor n0, growth rate g, 
        and time delay t0.
    
        Parameters
        ----------
            - t (float): The input variable of the function.
            - n0 (float): The scale factor of the logistic function.
            - g (float): The growth rate of the logistic function.
            - t0 (float): The time delay of the logistic function.
    
        Returns
        -------
            - f (float): The result of the logistic function.
        """
        
        # Mathematical calculations and returns a function value
        f = n0 / (1 + np.exp(-g*(t - t0)))
        return f


    def err_ranges(xdata, func, popt, sigma):
        
        """
        Calculates the upper and lower error ranges of a function.
    
        Args:
            - xdata (array-like): The input values for the function.
            - func (function): The function to be used for calculating 
                                the error ranges.
            - popt (array-like): The optimized parameters of the function.
            - sigma (float): The standard deviation of the error.
    
        Returns
        -------
            - err_low (array-like): The lower error range of the function.
            - err_up (array-like): The upper error range of the function.
        """
        
        # Mathematical calculations
        err_up = func(xdata, *popt + sigma)
        err_low = func(xdata, *popt - sigma)
        return err_low, err_up


    
    # Exponential fit plotting

    # Convert the `xlabel` column in `country_data` to a numeric type.
    country_data[xlabel] = pd.to_numeric(country_data[xlabel])

    # Fit an exponential curve to the data in the `xlabel` and `ylabel` columns
    # The `p0` argument specifies the initial values 
    # for the parameters of the curve.
    param, covar = opt.curve_fit(exponential, country_data[xlabel], 
                                 country_data[ylabel], p0=(1.2e12, 0.03))

    # Calculate the standard deviation of the parameters.
    sigma = np.sqrt(np.diag(covar))

    # Print the standard deviation.
    print(sigma)

    # Create a NumPy array of years from 1960 to 2030.
    year = np.arange(1960, 2030)

    # Calculate the forecast for the years in `year`.
    forecast = exponential(year, *param)

    # Calculate the upper and lower bounds of the error range for the forecast.
    low, up = err.err_ranges(year, exponential, param, sigma)

    # Add a column to `country_data` containing the fitted values.
    country_data["fit"] = exponential(country_data[xlabel], *param)

    # Create a figure.
    plt.figure()

    # Plot the data points.
    plt.plot(country_data[xlabel], country_data[ylabel], label=ylabel)

    # Plot the forecast.
    plt.plot(year, forecast, label="Forecast")

    # Fill the area between the forecast and 
    # the error range with a yellow fill.
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)

    # Set the x-axis label.
    plt.xlabel(xlabel)

    # Set the y-axis label.
    plt.ylabel(ylabel)

    # Set the title.
    # The `country` variable is used to dynamically set the title.
    plt.title("Exponential fit plot of {}".format(country))

    # Add a legend.
    plt.legend()

    # Show the plot.
    plt.show()

    
    
    # Polynomial fit plotting

    # Fit a polynomial curve to the data in the `xlabel` and `ylabel` columns.
    param, covar = opt.curve_fit(poly, country_data[xlabel], 
                                 country_data[ylabel])

    # Calculate the standard deviation of the parameters.
    sigma = np.sqrt(np.diag(covar))

    # Print the standard deviation.
    print(sigma)

    # Create a NumPy array of years from 1960 to 2030.
    year = np.arange(1960, 2030)

    # Calculate the forecast for the years in `year`.
    forecast = poly(year, *param)

    # Calculate the upper and lower bounds of the error range for the forecast.
    low, up = err.err_ranges(year, poly, param, sigma)

    # Add a column to `country_data` containing the fitted values.
    country_data["fit"] = poly(country_data[xlabel], *param)

    # Create a figure.
    plt.figure()

    # Plot the data points.
    plt.plot(country_data[xlabel], country_data[ylabel], label=ylabel)

    # Plot the forecast.
    plt.plot(year, forecast, label="Forecast")

    # Fill the area between the forecast and the error range with a yellow fill
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)

    # Set the x-axis label.
    plt.xlabel(xlabel)

    # Set the y-axis label.
    plt.ylabel(ylabel)

    # Set the title.
    # The `country` variable is used to dynamically set the title.
    plt.title("Polynomial fit plot of {}".format(country))

    # Add a legend.
    plt.legend()

    # Show the plot.
    plt.show()
    
    
    # Logistic fit plotting

    # Fit a logistic curve to the data in the `xlabel` and `ylabel` columns.
    # The `p0` argument specifies the initial values
    # for the parameters of the curve.
    param, covar = opt.curve_fit(logistic, country_data[xlabel], 
                                 country_data[ylabel], 
                                 p0=(1.2e12, 0.03, 1990.0))

    # Calculate the standard deviation of the parameters.
    sigma = np.sqrt(np.diag(covar))

    # Create a NumPy array of years from 1960 to 2030.
    year = np.arange(1960, 2030)

    # Calculate the forecast for the years in `year`.
    forecast = logistic(year, *param)

    # Calculate the upper and lower bounds of the error range for the forecast.
    low, up = err.err_ranges(year, logistic, param, sigma)

    # Plot the data points.
    plt.plot(country_data[xlabel], country_data[ylabel], label=ylabel)

    # Plot the forecast.
    plt.plot(year, forecast, label="Forecast")

    # Fill the area between the forecast and the error range with a yellow fill
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)

    # Set the x-axis label.
    plt.xlabel(xlabel)

    # Set the y-axis label.
    plt.ylabel(ylabel)

    # Set the title.
    # The `country` variable is used to dynamically set the title.
    plt.title("Logistic fit plot of {}".format(country))

    # Add a legend.
    plt.legend()

    # Show the plot.
    plt.show()

    # Printing GDP for 2030

    # Calculate the GDP forecast for 2030.
    forecast_data = logistic(2030, *param) / 1e9

    # Calculate the upper and lower bounds of the 
    # error range for the forecast for 2030.
    low, up = err.err_ranges(2030, logistic, param, sigma)

    # Calculate the standard deviation of the error 
    # range for the forecast for 2030.
    sig = np.abs(up - low) / (2.0 * 1e9)

    # Create the print statement with dynamic variables.
    print("{} value from 2030 forecast of {} is".format(ylabel, country), 
          forecast_data, "+/-", sig)
    


# Reading all required indicators

# Indicator name: Access to Electricity(% of Population)
ate, ate_t = read_world_bank_csv('Access to Electricity(% of Population).csv')

# Indicator name: Urban population
urban_pop, urban_pop_t = read_world_bank_csv('Urban population.csv')

# Indicator name: Methane emissions (kt of CO2 equivalent)
methane, methane_t = read_world_bank_csv(
    'Methane emissions (kt of CO2 equivalent).csv')

# Indicator name: Forest area (% of land area)
forest_area, forestarea_t = read_world_bank_csv(
    'Forest area (% of land area).csv')

# Indicator name: GDP growth(annual %)
gdp, gdp_t = read_world_bank_csv('GDP growth(annual %).csv')

# Indicator name: CO2 emmisons(kt)
ce, ce_t = read_world_bank_csv('CO2 emmisons(kt).csv')

# Indicator name: Agricultural land (% of land area)
agri, agri_t = read_world_bank_csv('Agricultural land (% of land area).csv')


# Indicator Name: School enrollment, primary and secondary (gross), (GPI)
gpi, gpi_t = read_world_bank_csv('GPI.csv')


# Correlation Analysis for several years by calling correlation_heatmap func.
# Saving figures 

correlation_heatmap('1990')
plt.savefig('Correlation_Heatmap_1990.png', dpi = 300)

correlation_heatmap('1999')
plt.savefig('Correlation_Heatmap_1999.png', dpi = 300)

correlation_heatmap('2009')
plt.savefig('Correlation_Heatmap_2009.png', dpi = 300)

correlation_heatmap('2019')
plt.savefig('Correlation_Heatmap_2019.png', dpi = 300)



# Cluster forming and analysis for various indicators
form_cluster(urban_pop, "Urban population") 
form_cluster(methane, "Methane emissions")    
analyze_clusters(methane, urban_pop, "Methane emissions", "Urban population" )

form_cluster(agri, "Agricultural land") 
form_cluster(agri, "Forest area")    
analyze_clusters(forest_area, agri, "Forest area", "Agricultural land" )



# Fitting analysis on different indicators and countries in top ten of GDP list

# Indicator name: GDP per capita growth (annual %)


gdppc, gdppc_t = read_world_bank_csv('GDP per capita growth (annual %).csv')
print('\nGDP per capita growth (annual %)')
print(gdp.head())
print(gdp_t.head())
print(gdp.describe())

# United States

gdp_usa = get_country_data(gdppc_t, 'United States', 'GDP')
country_data_analysis_plot(gdp_usa, 'Year', 'GDP', 'United States')

# China

gdp_china = get_country_data(gdp_t, 'China', 'GDP')
country_data_analysis_plot(gdp_china, 'Year', 'GDP', 'China')

# United Kingdom

gdp_uk = get_country_data(gdp_t, 'United Kingdom', 'GDP')
country_data_analysis_plot(gdp_uk, 'Year', 'GDP', 'United Kingdom')

# India

gdp_india = get_country_data(gdp_t, 'India', 'GDP')
country_data_analysis_plot(gdp_india, 'Year', 'GDP', 'India')


# Indicator name: GPI (Gender Parity Index)

# United States

gpi_usa = get_country_data(gpi_t, 'United States', 'GPI')
country_data_analysis_plot(gpi_usa, 'Year', 'GPI', 'United States')

# China

gpi_china = get_country_data(gpi_t, 'China', 'GPI')
country_data_analysis_plot(gpi_china, 'Year', 'GPI', 'China')

# United Kingdom

gpi_uk = get_country_data(gpi_t, 'United Kingdom', 'GPI')
country_data_analysis_plot(gpi_uk, 'Year', 'GPI', 'United Kingdom')

# India

gpi_india = get_country_data(gpi_t, 'India', 'GPI')
country_data_analysis_plot(gpi_india, 'Year', 'GPI', 'India')

# Indicator name: Forest area (% of land area)

# United States

fa_usa = get_country_data(forestarea_t, 'United States', 'Forest Area')
country_data_analysis_plot(fa_usa, 'Year', 'Forest Area', 'United States')

# China

fa_china = get_country_data(forestarea_t, 'China', 'Forest Area')
country_data_analysis_plot(fa_china, 'Year', 'Forest Area', 'China')

# United Kingdom

fa_uk = get_country_data(forestarea_t, 'United Kingdom', 'Forest Area')
country_data_analysis_plot(fa_uk, 'Year', 'Forest Area', 'United Kingdom')

# India

fa_india = get_country_data(forestarea_t, 'India', 'Forest Area')
country_data_analysis_plot(fa_india, 'Year', 'Forest Area', 'India')



