import numpy as np
import matplotlib.pyplot as plt
import spe_loader as sl
import scipy as sp
import scipy.constants as sc
from qdanalysis.spectra_analysis.spefile3 import *
import glob
from scipy.stats import norm
import os
import pandas as pd
from qdanalysis.spectra_analysis.spefile3 import speFile3

from IPython.display import display, clear_output
import ipywidgets as widgets
import tkinter as tk
from tkinter import filedialog

import time
from IPython.display import display, clear_output
import ipywidgets as widgets

def spec(filename, frameNum):
    s = speFile3(filename)
    s.quickPlot(frame = frameNum, ylim= None, units = 'nm')
    #s.specDiffusionPlot(maxs = None, units = 'nm')
    #s.plotFramesColor()
    return s 

if __name__ == '__main__':
    file_path = filedialog.askopenfilename(title="Select SPE file", filetypes=[("SPE files", "*.spe")])
    if not file_path:
        raise ValueError("No file selected")
    frameNum = 0 
    spec(file_path, frameNum)
    # plt.xlim(613,615)
    # plt.ylim(0, 1000)
    plt.show()
    