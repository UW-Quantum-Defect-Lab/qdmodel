# This file provides a template for doing lifetime analysis. It will contatin
# core functionality needed, allowing you to just take it and add in your 
# own functionality for specific use cases.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import special

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Extract the column headers from the first line
        column_headers = lines[0].strip().split('\t')

        # Create a list to store the data
        data = []

        # Process each line in the text file (excluding the first line)
        for line in lines[1:]:
            # Split the line into columns using tabs as the delimiter
            columns = [int(i) for i in line.strip().split('\t')]

            # Append the columns to the data list
            data.append(columns)

        # Create a pandas dataframe from the data and set the column headers
        df = pd.DataFrame(data, columns=column_headers)

# reads in multiple files and sums the counts per bin across files
def read_multiple_text_files(file_paths):
    a = []
    b = []
    l = len(file_paths)
    # Process each file in the list 
    c = False
    for file_path in file_paths:
        d = read_text_file(file_path)
        if c == False:
            # a = d.values[180:, 0]/1000
            a = d.values[:, 0]/1000
            # b = d.values[180:, 1]
            b = d.values[:, 1]
            c = True
        else:
            # b += d.values[180:, 1]
            b += d.values[:, 1]
    return a,b,l

# function to draw graph, if you want multiple data sets drawn on the same graph
# set render equal to False on all your plot_data() calls up until the last one.
def plot_data(x, y, line_name, x_label, y_label, fmt_str='', render=True):
    # Refer to docs for more info on fmt -> https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.plot.html
    plt.plot(x, y, fmt_str, label=line_name) if fmt_str else plt.plot(x, y, label=line_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if (render):
        # sorry no time for proper implementation, if you want to save the graph to your pc
        # uncomment safefig line and hard code file path
        # plt.savefig("PATH")
        plt.show()

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_gaussian(x, y):
    
    p0 = [np.max(y), np.mean(x), np.std(x)]  # Initial guess for the parameters
    
    # Fit the Gaussian curve to the data
    popt, pcov = curve_fit(gaussian, x, y, p0=p0)
    return popt, pcov

def single_exponential(x, amplitude, lifetime):
    return amplitude * np.exp(-x/lifetime)

def fit_single_exponential(x, y, amplitude_0, lifetime_0):
    p0 = [amplitude_0, lifetime_0]
    popt, pcov = curve_fit(single_exponential, x, y, p0=p0, maxfev=10000, gtol = 1e-10)
    return popt, pcov

def bi_exponential(x, amplitude_1, lifetime_1, amplitude_2, lifetime_2):
    return amplitude_1 * np.exp(-x/lifetime_1) + amplitude_2 * np.exp(-x/lifetime_2)

def fit_biexponential(x, y, amplitude_1_0, lifetime_1_0, amplitude_2_0, lifetime_2_0):
    p0 = [amplitude_1_0, lifetime_1_0, amplitude_2_0, lifetime_2_0]    
    popt, pcov = curve_fit(bi_exponential, x, y, p0=p0, maxfev=10000, gtol = 1e-10)
    return popt, pcov

# a is tau1, b is tau2, c is amplitude of tau1, d is amplitude of tau2, f is the time offset of gaussian
def expgaussian(x, a, b, c, d, f):
    return c/2*np.exp(a/2*(2*f+a*0.449**2-2*x))*special.erfc((f+a*0.449**2-x)/(np.sqrt(2)*0.449))+d/2*np.exp(b/2*(2*f+b*0.449**2-2*x))*special.erfc((f+b*0.449**2-x)/(np.sqrt(2)*0.449))
# https://arxiv.org/pdf/2201.03561.pdf
def fit_expgaussian(x,y):
    p0 = [0.1, 0.1, 1, 1, 20]  # Initial guess for the parameters
    boundsp0 = ((0,0, 0, 0, 10),(20,20,100,100, 30))
    popt, pcov = curve_fit(expgaussian, x, y, p0=p0, bounds = boundsp0, maxfev=10000, gtol = 1e-10)
    return popt, pcov

# Just going to keep these as is, don't want to mess with them too much, so if you want to use them
# it won't be as plug and play. The 0.449 is a hard coded std dev