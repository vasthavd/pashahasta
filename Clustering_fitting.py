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




def get_country_data(df, country, value):
    
    """
    Returns a pandas DataFrame containing the data for a given country and a specific value.

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
        A DataFrame containing the year and the specified value for the given country.
    """
    country_data = df.loc[:,[country]].reset_index().dropna()
    country_data.columns = ['Year', value]
    return country_data



# Indicator name: Access to Electricity(% of Population)

ate, ate_t = read_world_bank_csv('GPI.csv')
print('\nAccess to Electricity(% of Population)')
print(ate.head())
print(ate_t.head())
print(ate.describe())


gam = get_country_data(ate_t, 'India', 'GPI')


def country_data_analysis_plot(country_data, xlabel, ylabel, country):
    
    """
    Creates and displays plots of different fit functions for a given country's data.

    Args:
        country_data (pandas.DataFrame): A DataFrame containing the country's data.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        country (str): The name of the country.

    Returns:
        None
    """
    
    def exponential(t, n0, g):
        
        """
        Calculates an exponential function with scale factor n0 and growth rate g.
    
        Args:
            - t (float): The input variable of the function.
            - n0 (float): The scale factor of the exponential function.
            - g (float): The growth rate of the exponential function.
    
        Returns:
            - f (float): The result of the exponential function.
        """
        t = t - 1990
        f = n0 * np.exp(g*t)
        return f


    def poly(x, a, b, c, d, e):
        
        """
        Calculates a polynomial function.
    
        Args:
            - x (float): The input variable of the function.
            - a (float): The constant term of the polynomial.
            - b (float): The coefficient of x.
            - c (float): The coefficient of x^2.
            - d (float): The coefficient of x^3.
            - e (float): The coefficient of x^4.
    
        Returns:
            - f (float): The result of the polynomial function.
        """
        x = x - 1990
        f = a + b*x + c*x**2 + d*x**3 + e*x**4
        return f


    def logistic(t, n0, g, t0):
        """
        Calculates a logistic function with scale factor n0, growth rate g, and time delay t0.
    
        Args:
            - t (float): The input variable of the function.
            - n0 (float): The scale factor of the logistic function.
            - g (float): The growth rate of the logistic function.
            - t0 (float): The time delay of the logistic function.
    
        Returns:
            - f (float): The result of the logistic function.
        """
        f = n0 / (1 + np.exp(-g*(t - t0)))
        return f


    def err_ranges(xdata, func, popt, sigma):
        """
        Calculates the upper and lower error ranges of a function.
    
        Args:
            - xdata (array-like): The input values for the function.
            - func (function): The function to be used for calculating the error ranges.
            - popt (array-like): The optimized parameters of the function.
            - sigma (float): The standard deviation of the error.
    
        Returns:
            - err_low (array-like): The lower error range of the function.
            - err_up (array-like): The upper error range of the function.
        """
        err_up = func(xdata, *popt + sigma)
        err_low = func(xdata, *popt - sigma)
        return err_low, err_up



    country_data[xlabel] = pd.to_numeric(country_data[xlabel])
    
    # Exponential fit
    #print(type(country_data["Year"].iloc[1]))
    country_data[xlabel] = pd.to_numeric(country_data[xlabel])
    print(type(country_data[xlabel].iloc[1]))
    param, covar = opt.curve_fit(exponential, country_data[xlabel], country_data[ylabel], p0=(1.2e12, 0.03))
    sigma = np.sqrt(np.diag(covar))
    print(sigma)
    year = np.arange(1960, 2031)
    forecast = exponential(year, *param)
    low, up = err.err_ranges(year, exponential, param, sigma)
    country_data["fit"] = exponential(country_data[xlabel], *param)
    plt.figure()
    plt.plot(country_data[xlabel], country_data[ylabel], label=ylabel)
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # set the title with dynamic label values
    plt.title("Exponential fit plot of {} vs. {} for {}".format(ylabel, xlabel, country))
    plt.legend()
    plt.show()
    
    
    # Polynomial fit
    param, covar = opt.curve_fit(poly, country_data[xlabel], country_data[ylabel])
    sigma = np.sqrt(np.diag(covar))
    print(sigma)
    year = np.arange(1960, 2031)
    forecast = poly(year, *param)
    low, up = err.err_ranges(year, poly, param, sigma)
    country_data["fit"] = poly(country_data[xlabel], *param)
    plt.figure()
    plt.plot(country_data[xlabel], country_data[ylabel], label=ylabel)
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # set the title with dynamic label values
    plt.title("Polynomial fit plot of {} vs. {} for {}".format(ylabel, xlabel, country))
    plt.legend()
    plt.show()
    
    # Logistic fit
    param, covar = opt.curve_fit(logistic, country_data[xlabel], country_data[ylabel],
    p0=(1.2e12, 0.03, 1990.0))
    low, up = err.err_ranges(year, logistic, param, sigma)
    sigma = np.sqrt(np.diag(covar))
    plt.figure()
    plt.plot(country_data[xlabel], country_data[ylabel], label=ylabel)
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # set the title with dynamic label values
    plt.title("Logistic fit plot of {} vs. {} for {}".format(ylabel, xlabel, country))
    plt.legend()
    plt.show()
    
    # Printing GDP for 2030
    forecast_data = logistic(2030, *param)/1e9
    low, up = err_ranges(2030, logistic, param, sigma)
    sig = np.abs(up-low)/(2.0 * 1e9)
    print("{} value for 2030 for {} is", forecast_data, "+/-", sig)
    # create the print statement with dynamic variables
    print("{} value from 2030 forecast of {} is".format(ylabel, country), forecast_data, "+/-", sig)
    
    
country_data_analysis_plot(gam, 'Year', 'GPI', 'India')
