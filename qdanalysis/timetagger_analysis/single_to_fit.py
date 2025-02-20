# This code will be to take a ttbin file from the swabian time tagger which has a 42ps timing
# resolution and process it to get a unnormalized g2 dip. 
# Fitting based here: https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010202
# python version 3.10.11

# timetagger swabian ultra has a python api: https://www.swabianinstruments.com/static/documentation/TimeTagger/api/TimeTaggerLibrary.html
# THIS ONLY WORKS on WINDOWS and assumes you have installed the package so you can import the Timetagger module
# The package is not public and has to be installed with the whole damn software

import TimeTagger
import pandas as pd
from tkinter import filedialog
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from tqdm import tqdm
from blinking_analysis import *

# this class is to read in the timetagger ttbin files and get the raw timetags
class ttbindata:
    def __init__(self):
        files = filedialog.askopenfilenames() # ask for filenames of ttbin files
        self.filepath = files[0][:files[0].rfind('/')+1]
        filereader = TimeTagger.FileReader(files) 
        data = pd.DataFrame()
        while filereader.hasData():
            df = pd.DataFrame()
            t = filereader.getData(1000000)
            df['Timestamps'] = t.getTimestamps() # ps
            df['Channel'] = t.getChannels() 
            df['Event Type'] = t.getEventTypes() # https://www.swabianinstruments.com/static/documentation/TimeTagger/api/TimeTaggerLibrary.html#TagType
            data = pd.concat([data, df])

        # shift origin
        data['Timestamps'] = data['Timestamps'] - data.iloc[0][0]
        # checks for overflow
        if sum(data['Event Type'] != 0):
            raise Exception("Overflow happened. Check data")
        self.data = data

        self.ch1 = data.Channel.unique()[0]  # physical channel 1
        ch1_tt = data[data['Channel'] == self.ch1]['Timestamps'] # in ps

        # numpy faster than using pandas
        self.ch1np = ch1_tt.to_numpy() # timetags of channel 1

        # find last time over two channels
        maxTime = max(self.ch1np)
        self.maxTime = int(math.ceil(maxTime/1e12)*1e12) # last time recorded in ns
        
    # plots avg counts over time
    def plotAvgCounts(self, savename = "", xlim=None):
        # group into bins of 100 milliseconds
        seconds = []
        for i in range(0, self.maxTime, int(1e11)):
            seconds.append(i)
        s1 = np.histogram(self.ch1np,seconds)[0]
        seconds.pop(0)
        mins = [x/60/1e12 for x in seconds]

        # for multiple files have to delete places when not taking data i.e. counts = 0
        s1z = np.where(s1 == 0)[0]
        s1 = np.delete(s1,s1z)*10 # have to multiply by 10 because bins of 100ms
        min1 = np.delete(mins, s1z)
        self.seconds = np.delete(seconds,s1z) # for splitting the dark and bright states
        self.s1 = s1
        self.min1 = min1

        self.avg1 = np.mean(s1)

        # plot in terms of minutes
        plt.plot(min1, s1, label = 'Channel '+ str(self.ch1), color='coral')
        plt.xlabel("Time (min)")
        plt.ylabel("Counts/sec")
        plt.title('Average counts/sec for ' + str(round(self.maxTime*1e-12/60)) + " mins")
        plt.legend()

        if xlim:
            plt.xlim(xlim)

        plt.savefig(self.filepath+'avgCounts'+savename)
        plt.show()

    # fit a exponential fit
    def fitExp(self, data, min1):
        def func(x, a, b, c):
            return a*np.exp(-x/b) + c
        popt, _ = curve_fit(func, min1, data/max(data))
        print(popt)
        return func(min1, *popt), popt[1]
    
    # plots avg counts with fit over time
    def plotAvgCountsFit(self, savename = "", xlim=None):
        # group into bins of 100 milliseconds
        seconds = []
        for i in range(0, self.maxTime, int(1e11)):
            seconds.append(i)
        s1 = np.histogram(self.ch1np,seconds)[0]
        seconds.pop(0)
        mins = [x/60/1e12 for x in seconds]

        # for multiple files have to delete places when not taking data i.e. counts = 0
        s1z = np.where(s1 == 0)[0]
        s1 = np.delete(s1,s1z)*10 # have to multiply by 10 because bins of 100ms
        min1 = np.delete(mins, s1z)
        self.seconds = np.delete(seconds,s1z) # for splitting the dark and bright states
        self.s1 = s1
        self.min1 = min1

        self.avg1 = np.mean(s1)

        fitted_data, t1 = self.fitExp(s1, min1)

        # plot in terms of minutes
        plt.plot(min1, s1/max(s1), label = 'Channel '+ str(self.ch1))
        plt.plot(min1, fitted_data, label = 'Exponential fit with t1 = ' + str(round(t1,2)) + ' mins')
        plt.xlabel("Time (min)")
        plt.ylabel("Normalized Counts/sec")
        plt.title('Average counts/sec for ' + str(round(self.maxTime*1e-12/60)) + " mins")
        plt.legend()

        if xlim:
            plt.xlim(xlim)

        plt.savefig(self.filepath+'avgCountsFit'+savename)
        plt.show()

def main():
    t  = ttbindata()
    t.plotAvgCounts()
    t.plotAvgCountsFit()

if __name__ == "__main__":
    main()