import numpy as np
import matplotlib.pyplot as plt
import spe_loader as sl
import scipy as sp
import scipy.constants as sc
from spefile3 import *
import glob
from scipy.stats import norm
import os
import pandas as pd
from spefile3 import speFile3

from IPython.display import display, clear_output
import ipywidgets as widgets
import tkinter as tk
from tkinter import filedialog

import time
from IPython.display import display, clear_output
import ipywidgets as widgets

def plot_all_frames(spe_file):
    """
    Loops through all frames of the given speFile3 object and plots them with a 1 second delay between frames.
    Allows pausing the loop at a given frame with a pause button.
    
    Parameters:
    spe_file (speFile3): The speFile3 object containing the data.
    """
    num_frames = spe_file.numFrames
    wavelengths = spe_file.wavelengths
    
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    paused = False
    
    def on_pause_button_clicked(b):
        nonlocal paused
        paused = not paused
        if paused:
            b.description = 'Resume'
        else:
            b.description = 'Pause'
    
    pause_button = widgets.Button(description='Pause')
    pause_button.on_click(on_pause_button_clicked)
    
    # Create a new widget box to hold the plot and the button
    output = widgets.Output()
    display(widgets.VBox([output, pause_button]))
    
    for frame in range(num_frames):
        while paused:
            plt.pause(0.1)
        
        data = spe_file.data[frame][0][0]
        with output:
            ax.clear()
            ax.plot(wavelengths, data, label=f'Frame {frame + 1}')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Counts')
            ax.set_title('All Frames')
            ax.legend()
            display(fig)
            clear_output(wait=True)
            plt.pause(1)  # 1 second delay between frames

    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == '__main__':
    # Load the data
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select SPE file", filetypes=[("SPE files", "*.spe")])
    if not file_path:
        raise ValueError("No file selected")
    spe_file = speFile3(file_path)
    plot_all_frames(spe_file)