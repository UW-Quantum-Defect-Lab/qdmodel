# This file is to look at g2 for the bright and dark states for a blinking dot

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from timetagger_to_fit import *

# algothorim to find the optimal split for the bright and dark states
# https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization
def get_jenks_breaks(data_list2, number_class):
    data_list = np.copy(data_list2)
    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_class + 1):
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1
    return kclass

# plotting jen breaks overlay with avg counts
def plotJenks(t, breaks): 
    plt.plot(t.min1, t.s1)
    for line in breaks:
        plt.axhline(y = line, color = 'k', linestyle = '--')
    plt.xlabel("Time (min)")
    plt.ylabel("Counts/sec")
    plt.grid(True)
    plt.savefig(t.filepath+'avgCountsJens')
    plt.show()

# this is for splitting the time tag data into the bright and dark states
def split_timetags(ttbindata, breaks):
    brightCounts1 = []
    brightCounts2 = []
    dimCounts1 = []
    dimCounts2 = []
    index1 = 0
    index2 = 0
    l1 = len(ttbindata.ch1np)
    l2 = len(ttbindata.ch2np)
    for i in range(0, len(ttbindata.seconds)):
        if ttbindata.s1[i] >= breaks[1]:
            while index1 < l1 and ttbindata.ch1np[index1] < ttbindata.seconds[i]:
                brightCounts1.append(ttbindata.ch1np[index1])
                index1 += 1
            while index2 < l2 and ttbindata.ch2np[index2] < ttbindata.seconds[i]:
                brightCounts2.append(ttbindata.ch2np[index2])
                index2 += 1
        else:
            while index1 < l1 and ttbindata.ch1np[index1] < ttbindata.seconds[i]:
                dimCounts1.append(ttbindata.ch1np[index1])
                index1 += 1
            while index2 < l2 and ttbindata.ch2np[index2] < ttbindata.seconds[i]:
                dimCounts2.append(ttbindata.ch2np[index2])
                index2 += 1
    return brightCounts1,brightCounts2,dimCounts1,dimCounts2

def plot_split_tt(tt,c1,c2):
    # group into bins of 100 milliseconds
    seconds = []
    for i in range(0, tt.maxTime, int(1e11)):
        seconds.append(i)
    s1 = np.histogram(c1,seconds)[0]*10
    s2 = np.histogram(c2,seconds)[0]*10

    # plot in terms of minutes
    seconds.pop(0)
    mins = [x/60/1e12 for x in seconds]
    plt.plot(mins, s1, label = 'Channel '+ str(tt.ch1))
    plt.plot(mins, s2, label = 'Channel ' + str(tt.ch2))
    plt.xlabel("Time (min)")
    plt.ylabel("Counts/sec")
    plt.title('Average counts/sec for ' + str(round(tt.maxTime*1e-12/60)) + " mins")
    plt.legend()
    plt.show()