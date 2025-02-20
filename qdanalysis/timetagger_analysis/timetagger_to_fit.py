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

        self.ch2 = data.Channel.unique()[0]  # physical channel 1
        self.ch1 = data.Channel.unique()[1]  # physical channel 2
        ch1_tt = data[data['Channel'] == self.ch1]['Timestamps'] # in ps
        ch2_tt = data[data['Channel'] == self.ch2]['Timestamps'] # in ps

        # numpy faster than using pandas
        self.ch1np = ch1_tt.to_numpy() # timetags of channel 1
        self.ch2np = ch2_tt.to_numpy() # time tags of channel 2 

        # find last time over two channels
        maxTime = 0
        if max(self.ch1np) > maxTime:
            maxTime = max(self.ch1np)
        if max(self.ch2np) > maxTime:
            maxTime = max(self.ch2np)
        self.maxTime = int(math.ceil(maxTime/1e12)*1e12) # last time recorded in ns
        
    # plots avg counts over time
    def plotAvgCounts(self, savename = ""):
        # group into bins of 100 milliseconds
        seconds = []
        for i in range(0, self.maxTime, int(1e11)):
            seconds.append(i)
        s1 = np.histogram(self.ch1np,seconds)[0]
        s2 = np.histogram(self.ch2np,seconds)[0]
        seconds.pop(0)
        mins = [x/60/1e12 for x in seconds]

        # for multiple files have to delete places when not taking data i.e. counts = 0
        s1z = np.where(s1 == 0)[0]
        s2z = np.where(s2 == 0)[0]
        s1 = np.delete(s1,s1z)*10 # have to multiply by 10 because bins of 100ms
        s2 = np.delete(s2, s2z)*10
        min1 = np.delete(mins, s1z)
        min2 = np.delete(mins, s2z)
        self.seconds = np.delete(seconds,s1z) # for splitting the dark and bright states
        self.s1 = s1
        self.min1 = min1

        self.avg1 = np.mean(s1)
        self.avg2 = np.mean(s2)

        # plot in terms of minutes
        plt.plot(min1, s1, label = 'Channel '+ str(self.ch1))
        plt.plot(min2, s2, label = 'Channel ' + str(self.ch2))
        plt.xlabel("Time (min)")
        plt.ylabel("Counts/sec")
        plt.title('Average counts/sec for ' + str(round(self.maxTime*1e-12/60)) + " mins")
        plt.legend()
        plt.savefig(self.filepath+'avgCounts'+savename)
        plt.show()

# this class is for binning data for cross-correlation g2
class g2bins:
    def __init__(self, ttfile, binsize = 1e-9, window = np.array([-1, 1])*200e-9, name = '') -> None:
        self.binps = binsize/1e-12 # in ps
        self.winps = window/1e-12 # in ps
        timeDiffs = []
        kloopmin = 0
        kloopmax = len(ttfile.ch2np)
        self.ttfile = ttfile
        self.name = name

        # binning the data (tqdm is for progress bar)
        for j in tqdm(range(len(ttfile.ch1np))):
            for k in range(kloopmin, kloopmax):
                tD = ttfile.ch2np[k] - ttfile.ch1np[j]

                # this is for speeding up the binning
                # checks if difference is within the window specified for speedup
                if tD <= self.winps[0]:
                    kloopmin = k # only have to look at larger time diffs
                elif tD >= self.winps[1]:
                    break # only have to look at smaller time diffs
                else:
                    timeDiffs.append(tD)
        self.timeDiffs = timeDiffs 

    # center time is where the g2(0) should be
    def unnormalized_g2(self, centertime = 0, save = False):
        # plotting the histogram of the binned data unnormalized
        counts,bins = np.histogram(self.timeDiffs, bins = int((self.winps[1] - self.winps[0])/self.binps), 
                                   range = (self.winps[0], self.winps[1]))
        bins = bins*1e-3 # to ns
        self.center = centertime 
        bincenter = [(bins[i + 1] + bins[i])/2 for i in range(len(bins)-1)] # get the average value of the bin 
        bincenter = np.array(bincenter) + self.center # g2(0) should be at 38 ns
        plt.plot(bincenter, counts, color = 'mediumpurple')
        plt.ylabel("Coincidences/bin")
        plt.xlabel(r'$\tau$ (ns)')
        plt.ylim([min(counts)*0.9, max(counts)*1.11])
        plt.rcParams.update({'font.size':14})
        if save:
            plt.savefig(self.ttfile.filepath+"unnormalized"+self.name)
        plt.show()
        self.counts = counts
        self.bincenter = bincenter 

    # normalization by doing g2/g2(inf)
    def normWingsG2(self, save = False):
        totalTime = self.ttfile.maxTime*1e-12 # in s
        cn = self.counts/(self.ttfile.avg1*self.ttfile.avg2*totalTime*self.binps*1e-12)
        g2 = cn/np.mean(cn[1:50])
        plt.plot(self.bincenter, g2)
        plt.ylabel(r"$g^{(2)}(\tau)$")
        plt.xlabel(r'$\tau$ (ns)')
        plt.ylim([min(g2)*0.9, max(g2)*1.11])
        plt.rcParams.update({'font.size':14})
        plt.title(r"$g^{(2)}(\tau)/g^{(2)}(\infty)$ - " + str(round(self.ttfile.maxTime*1e-12/60)) + " mins")
        if save:
            plt.savefig(self.ttfile.filepath+"normalizedg2wings"+self.name)
        plt.show() 

    # normalization with the background
    def normalized_g2(self, avgbk1 = 300, avgbk2 = 400, save = False):
        ## normalization based on https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010202#:~:text=Photon%2Demission%2Dcorrelation%20spectroscopy%20is,key%20property%20for%20quantum%20technology.
        # dont need to do timing jitter correction since my rms gaussian timing jitter is 54.3 ps << lifetime of emitter
        totalTime = self.ttfile.maxTime*1e-12 # in s
        avg1 = self.ttfile.avg1
        avg2 = self.ttfile.avg2
        
        # making sure the higher count channel is in channel 1
        if avg2 > avg1:
            avg1, avg2 = avg2, avg1
            avgbk1, avgbk2 = avgbk2, avgbk1

        # since the beam splitter is about 41.6/58.4, the normalization is not trivial
        # Also, since the microscope drifts, the avg count rate decreases over time so have to optimize
        # over the count rate to get a g2 at the wings of 1
        def avgmin(x):
            a1 = avg1 + avg1*x/(avg1+avg2)
            a2 = avg2 + avg2*x/(avg1+avg2)
            cn = self.counts/(a1*a2*totalTime*self.binps*1e-12)
            rho1 = a1/(a1 + avgbk1)
            rho2 = a2/(a2 + avgbk2)
            g2 = (cn + rho1*rho2 - 1)/(rho1*rho2)
            return abs(np.mean(g2[1:50])-1)
        resd = minimize(avgmin, 0, options={'disp': True})

        # taking optimal counts and plotting the g2 
        a1 = avg1 + avg1*resd.x/(avg1+avg2)
        a2 = avg2 + avg2*resd.x/(avg1+avg2)
        cn = self.counts/(a1*a2*totalTime*self.binps*1e-12)
        rho1 = a1/(a1 + avgbk1)
        rho2 = a2/(a2 + avgbk2)
        g2 = (cn + rho1*rho2 - 1)/(rho1*rho2)
        self.g2 = g2 #- 0.1
        save_this = np.array([self.bincenter, self.g2])
        np.savetxt(self.ttfile.filepath+"g2"+self.name+".csv", save_this, delimiter = ',')
        plt.plot(self.bincenter, self.g2)
        plt.ylabel(r"$g^{(2)}(\tau)$")
        plt.xlabel(r'$\tau$ (ns)')
        plt.ylim([min(self.g2)*0.9, max(self.g2)*1.11])
        plt.rcParams.update({'font.size':14})
        plt.title(r"$g^{(2)}(\tau)$ - " + str(round(self.ttfile.maxTime*1e-12/60)) + " mins")
        if save:
            plt.savefig(self.ttfile.filepath+"normalizedg2"+self.name)
        plt.show() 

    # fits the g2 data. Might have to modify the fitting parameters for better fits
    def g2fit(self, fitcase = "single", save = False):
        minindex = (np.abs(self.bincenter)).argmin()+2
        shortbin = self.bincenter[:minindex]
        shortg2 = self.g2[:minindex]
        shortbin = shortbin*-1
        shortbin = shortbin - min(shortbin)

        def fitg2(fitcase):
            match fitcase:
                case "double": # double time scale case 
                    def double(x, a, b, c, d):
                        return 1 - a*np.exp(-x/b) + c*np.exp(-x/d)
                    parameters, _ = curve_fit(double, shortbin, shortg2, p0 = [0.1,12,0.1,100], bounds = ((-10, .1,-10,.1),(10,100,10,1000)))
                    fit = double(shortbin, *parameters)
                    return fit, parameters 
                case default: # single case
                    def single(x, a, b):
                        return 1 - a*np.exp(-x/b)
                    parameters, _ = curve_fit(single, shortbin, shortg2, p0 = [0.1,12], bounds = ((-1000, 0.01),(100,1000)))
                    fit = single(shortbin, *parameters)
                    return fit, parameters
                
        fittg2, parameters = fitg2(fitcase)
        plt.plot(self.bincenter, self.g2)
        if len(parameters) > 2:
            plt.plot(shortbin, fittg2, color = 'red', label = r"$\tau_1$ = "+str(round(parameters[1],2)) + r"ns, $\tau_2$ = " + str(round(parameters[3],2))+"ns")
        else:
            plt.plot(shortbin, fittg2, color = 'red', label = r"$\tau_1$ = "+str(round(parameters[1],2))+"ns")
            save_this = np.array([shortbin, fittg2])
            np.savetxt(self.ttfile.filepath+"g2fit"+self.name+".csv", save_this, delimiter = ',')
        plt.plot(-1*shortbin, fittg2, color = 'red')
        plt.xlim([self.bincenter[0], self.bincenter[-1]])
        plt.ylabel(r"$g^{(2)}(\tau)$")
        plt.xlabel(r'$\tau$ (ns)')
        plt.ylim([min(self.g2)*0.9, max(self.g2)*1.11])
        plt.rcParams.update({'font.size':14})
        plt.title(r"$g^{(2)}(\tau)$ - " + str(round(self.ttfile.maxTime*1e-12/60)) + " mins")
        plt.legend(frameon = False)
        if save:
            plt.savefig(self.ttfile.filepath+'g2fit'+self.name)
        plt.show()
        print("fit params: " + str(parameters)) 

def g2exp(t, binsize, window, center, bkgC, save = False, fitcase = "single", name = ''):
    g2 = g2bins(t, binsize, window, name)
    g2.unnormalized_g2(center, save) 
    g2.normWingsG2(save)
    g2.normalized_g2(bkgC, bkgC, save)
    g2.g2fit(fitcase, save) 

# binsize in sec, win in sec, centerTime in ns, bkgcounts in c/s 
def experiment(binsize, win, center, bkgC, save = False, fitcase = "single"):
    t  = ttbindata()
    t.plotAvgCounts()
    window = np.array([-1, 1])*win 
    g2exp(t, binsize, window, center, bkgC, save, fitcase, name = '')

# for analyzing blinking and splitting the data set into bright and dark counts
def blinkingExperiment(binsize, win, center, bkgC, save = False):
    window = np.array([-1, 1])*win 
    # normal
    t  = ttbindata()
    t.plotAvgCounts()
    g2exp(t, binsize, window, center, bkgC, save, name = '')

    # divide data
    breaks = get_jenks_breaks(t.s1, 2) # 2 is the number of jen breaks 
    #breaks[1] = 12500
    plotJenks(t, breaks)
    b1, b2, d1, d2 = split_timetags(t, breaks)

    # bright exp
    plot_split_tt(t, b1,b2)
    t.ch1np = b1
    t.ch2np = b2
    g2exp(t, binsize, window, center, bkgC, save, name = 'bright')

    # dark exp
    plot_split_tt(t, d1, d2)
    t.ch1np = d1
    t.ch2np = d2
    g2exp(t, binsize, window, center, bkgC, save, name = 'dark')

# does g2 for a window in min given by timeStart and timeEnd
def filteredExperiment(timeStart, timeEnd, binsize, win, center, bkgC, save = False):
    window = np.array([-1, 1])*win 
    # normal
    t  = ttbindata()
    t.plotAvgCounts()
    g2exp(t, binsize, window, center, bkgC, save, name = '')

    # doing the filtering 
    filteredcounts1 = []
    filteredcounts2 = []
    index1 = 0
    index2 = 0
    l1 = len(t.ch1np)
    l2 = len(t.ch2np)
    for i in range(0, len(t.min1)):
        if t.min1[i] >= timeStart and t.min1[i] <= timeEnd:
            while index1 < l1 and t.ch1np[index1] < t.seconds[i]:
                filteredcounts1.append(t.ch1np[index1])
                index1 += 1
            while index2 < l2 and t.ch2np[index2] < t.seconds[i]:
                filteredcounts2.append(t.ch2np[index2])
                index2 += 1
        while index1 < l1 and t.ch1np[index1] < t.seconds[i]:
            index1 += 1
        while index2 < l2 and t.ch2np[index2] < t.seconds[i]:
            index2 += 1
    t.ch1np = filteredcounts1
    t.ch2np = filteredcounts2
    t.maxTime = int(math.ceil(max(t.ch1np)/1e12)*1e12)
    plot_split_tt(t, filteredcounts1, filteredcounts2)
    g2exp(t, binsize, window, center, bkgC, save, name = 'filtered')

def main():
    experiment(2e-9, 200e-9, 77, 2000, True, fitcase = "single")
    #blinkingExperiment(1e-9, 250e-9, 74, 400, False)
    #filteredExperiment(0, 10, 10e-9, 250e-9,0, 500, False)

if __name__ == "__main__":
         main()