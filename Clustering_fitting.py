# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:45:38 2023

@author: vasth
"""

# Importing all required modules with genarlised aliasing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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



# Defining a function group_years_by_count
def group_years_by_count(year_as_column_df):
    
    """
    Group years based on non-null count.

    This function takes a pandas dataframe as input, and creates a new    
    dataframe with non-null count and year columns. The years are then grouped
    into two categories based on whether their non-null count is above or below 
    the median count. The resulting groups are printed with a message 
    indicating the count, which is useful to determine years selection for 
    analysis.

    Parameters
    ----------
    year_as_column_df : pandas DataFrame
    A dataframe with one column containing year data.

    Returns
    -------
    result: str
    A string with two sections separated by a blank line. The first section
    lists the years with non-null count above the median count, and the
    second section lists the years with non-null count less than or equal
    to the median count.
    
    """
    
    # Create dataframe with non-null count and year columns
    non_null_years = year_as_column_df.notnull().sum()\
        .sort_values(ascending = False).to_frame().reset_index()
    non_null_years.columns = ['Year', 'Non-null Count']

    # Calculate median non-null count
    median_count = non_null_years['Non-null Count'].median()

    # Group years based on non-null count above or below median
    non_null_groups = non_null_years.groupby(
        non_null_years['Non-null Count'] > median_count)\
        .apply(lambda x: x['Year'].tolist())

    # Print groups with messages indicating count
    result = []
    result.append(
        f"Years with non-null count above the median count ({median_count}):")
    result.append(str(non_null_groups[True]))
    result.append("")
    result.append(
        f"Years with non-null count <= median count ({median_count}):")
    result.append(str(non_null_groups[False]))
    return '\n'.join(result)


# Defining a function line_plot
def line_plot(year_as_column_df, xlabel, ylabel, title):
    
  """
  Plots a line plot of the given DataFrame.

  Parameters
  ----------
    year_as_column_df: pandas DataFrame
    The DataFrame to plot.
    xlabel: str
    The label for the x-axis.
    ylabel: str
    The label for the y-axis.
    title: str
    The title for the plot.

  Returns
  -------
  None, Plots a matplotlib figure object: Line - Plot.
  
  """
  
  # Select the 10 countries to plot in order of the value list.
  countries_list = ['India']

  # Get the data for these countries.
  sorted_countries_list = year_as_column_df.loc[countries_list]

  # Plot the data.
  plt.figure()
  for country in countries_list:
    plt.plot(year_as_column_df.columns, 
             sorted_countries_list.loc[country], label = country)
    
  # Set the x-axis limits to the minimum and maximum years in the DataFrame.
  plt.xlim(min(year_as_column_df.columns), max(year_as_column_df.columns))
  
  # Set the x-axis ticks to the years, spaced every 5 years.
  plt.xticks(np.arange(0, len(year_as_column_df.columns)+1, 5), rotation = 90)

  # Set the label for the x-axis.    
  plt.xlabel(xlabel)
  
  # Set the label for the y-axis.
  plt.ylabel(ylabel)
  
  # Set the title for the plot.
  plt.title(title)
  
  
  # Add a legend to the plot, being labeled with the corresponding country.
  plt.legend(bbox_to_anchor = (1.05, 1))
  
  # Return the current figure object.
  return plt.gcf()


def get_country_data(df, country):
    country_data = df.loc[:,[country]].reset_index().dropna()
    country_data.columns = ['Year', 'value']
    return country_data


# Indicator name: Access to Electricity(% of Population)

ate, ate_t = read_world_bank_csv('Access to Electricity(% of Population).csv')
print('\nAccess to Electricity(% of Population)')
print(ate.head())
print(ate_t.head())
print(ate.describe())
print(group_years_by_count(ate))

india_data = get_country_data(ate_t, 'India')

def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f

# Select the "Year" and "value" columns from the India data dataframe
x = india_data["Year"]
y = india_data["value"]

param, covar = opt.curve_fit(poly, x, y)
sigma = np.sqrt(np.diag(covar))
print(sigma)
year = np.arange(1960, 2031)
forecast = poly(year, *param)

# Use the "Year" and "fit" columns from the India data dataframe
india_data["fit"] = poly(india_data["Year"].astype(int), *param)
plt.figure()
plt.plot(india_data["Year"], india_data["value"], label="value")
plt.plot(year, forecast, label="forecast")

low, up = err.err_ranges(year, poly, param, sigma)
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("value")
plt.legend()
plt.show()
