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
    


# Fitting analysis on different indicators and countries in top ten of GDP list

# Indicator name: GDP per capita growth (annual %)


gdp, gdp_t = read_world_bank_csv('GDP per capita growth (annual %).csv')
print('\nGDP per capita growth (annual %)')
print(gdp.head())
print(gdp_t.head())
print(gdp.describe())

# United States

gdp_usa = get_country_data(gdp_t, 'United States', 'GDP')
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

gpi, gpi_t = read_world_bank_csv('GPI.csv')

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

fa, fa_t = read_world_bank_csv('Forest area (% of land area).csv')

# United States

fa_usa = get_country_data(fa_t, 'United States', 'Forest Area')
country_data_analysis_plot(fa_usa, 'Year', 'Forest Area', 'United States')

# China

fa_china = get_country_data(fa_t, 'China', 'Forest Area')
country_data_analysis_plot(fa_china, 'Year', 'Forest Area', 'China')

# United Kingdom

fa_uk = get_country_data(fa_t, 'United Kingdom', 'Forest Area')
country_data_analysis_plot(fa_uk, 'Year', 'Forest Area', 'United Kingdom')

# India

fa_india = get_country_data(fa_t, 'India', 'Forest Area')
country_data_analysis_plot(fa_india, 'Year', 'Forest Area', 'India')



