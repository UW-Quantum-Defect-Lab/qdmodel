#This file is to generate a class to analyze spe 3.0 files. 
#Package to use 3.x Princeton Instruments SPE files is here: https://pypi.org/project/spe2py/#description 
# and https://github.com/ashirsch/spe2py and this can be installed with "pip install spe2py". 
# Make sure you have the right python version (3.11) and are using a virtual environment with that interpreter.

# Imports and initializaion
import numpy as np
import matplotlib.pyplot as plt
import spe_loader as sl
import scipy as sp
import scipy.constants as sc
import xmltodict as xmltodict
import re
from scipy.optimize import curve_fit
import os
import pandas as pd

h = sc.h # plank's constant in J*s
c = sc.c # speed of light in m/s
eV = sc.e # how many joules in 1eV


class speFile3:
    def __init__(self, filepath):
        # these are the data fields easily gotten by the downloaded spe2py
        data = sl.load_from_files([filepath])
        self.filepath = data.filepath
        self.wavelengths = data.wavelength
        self.data = data.data
        self.numFrames = len(self.data)

        # have to extract footer info myself. It's a really messy dict
        self.foot = self.__footer()

        # exposure time in seconds
        holder = self.foot['speformat.datahistories.datahistory.origin.experiment.devices.cameras.camera.shuttertiming.exposuretime._text']
        if holder == None:
            self.exposure = None
        else:
            self.exposure = int(holder)/1000.0 # in s

        # True if background corrected and False otherwise
        self.backgroundCorrected = self.stringtoBool(self.foot['speformat.datahistories.datahistory.origin.experiment.devices.cameras.camera.experiment.onlinecorrections.backgroundcorrection.enabled._text'])

        # file for background if corrected
        self.backgroundFile = self.foot['speformat.datahistories.datahistory.origin.experiment.devices.cameras.camera.experiment.onlinecorrections.backgroundcorrection.referencefile._text']
        
        # true if cosmic ray corrected applied
        self.cosmicRayCorrected = self.stringtoBool(self.foot['speformat.datahistories.datahistory.origin.experiment.devices.cameras.camera.experiment.onlinecorrections.cosmicraycorrection.enabled._text'])
        
        # grating size
        self.grating = self.foot['speformat.datahistories.datahistory.origin.experiment.devices.spectrometers.spectrometer.grating.selected._text']
        
        # date taken recorded by the spec i.e. date file created. This is not the actual date b/c computer date not configured to world clock time
        self.date  = self.foot['speformat.generalinformation.fileinformation._created']

    # for footer
    @staticmethod
    def str_to_valid_varname(string: str, ignore: str = '') -> str:
        """
        Convert a string to a valid Python variable name.

        Parameters
        ----------
        string : str
            The string to convert.
        ignore : str, optional
            A string containing characters to ignore during the conversion, by default ''.

        Returns
        -------
        str
            The converted string.

        Examples
        --------
        >>> str_to_valid_varname('a b c')
        'a_b_c'

        >>> str_to_valid_varname('a b.c', '.')
        'a_b.c'

        >>> str_to_valid_varname('a b,c.d', '.,')
        'a_b,c.d'
        """
        return re.sub(r'[^\w' + ignore + r']+|^(?=\d)', '_', string).lower()

    # For footer
    def __normalize_dict(self, dictionary: dict, parent_key='', separator='.') -> dict:
        """
        Normalize a dictionary by converting its keys to valid Python variable names and flattening it.
        Each key in the flattened dictionary is a result of addition of all keys that resulted in it from the
        original dictionary.

        For example, if the original dictionary is `{'a': {'b': 1, 'c': 2}, 'd': 3}`,
        the normalized dictionary is `{'a.b': 1, 'a.c': 2, 'd': 3}`, where '.' is the default separator.

        Parameters
        ----------
        dictionary : dict
            The dictionary to normalize.
        parent_key : str, optional
            The prefix to add to the keys of the dictionary, by default ''.
        separator : str, optional
            The string to use as a separator between the prefix and the keys, by default '.'.

        Returns
        -------
        Dict
            The normalized dictionary.

        Examples
        --------
        >>> normalize_dict({'a': {'b': 1, 'c': 2}, 'd': 3})
        {'a.b': 1, 'a.c': 2, 'd': 3}

        >>> normalize_dict({'a': {'b': 1, 'c': 2}, 'd': 3}, 'prefix')
        {'prefix.a.b': 1, 'prefix.a.c': 2, 'prefix.d': 3}

        >>> normalize_dict({'a': {'b': 1, 'c': 2}, 'd': 3}, 'prefix', '/')
        {'prefix/a/b': 1, 'prefix/a/c': 2, 'prefix/d': 3}
        """
        normalized_dict = {}
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            new_key = self.str_to_valid_varname(new_key, separator)
            if isinstance(value, dict):
                normalized_dict.update(self.__normalize_dict(value, new_key, separator))
            else:
                normalized_dict[new_key] = value
        return normalized_dict

    # extracts footer
    def __footer(self):
        file = open(self.filepath)
        footer_pos = sl.read_at(file, 678, 8, np.uint64)[0]
        file.seek(footer_pos)
        xmltext = file.read()
        norm_dict = self.__normalize_dict(xmltodict.parse(xmltext))
        return norm_dict
    
    @staticmethod
    def stringtoBool(str):
        if str == 'True':
            return True
        else:
            return False
        
    # Quick plots the data and if units is specified then plots it in eV. If multiple frames then plots the first frame as default
    def quickPlot(self, frame = 0, units = None, ylim = None):
        xaxis, xlab, _ = self.unitConversion(self.wavelengths, units = units)
        plt.plot(xaxis, self.data[frame][0][0])
        #plt.xlim([2, 2.1])
        plt.xlabel(xlab)
        plt.ylabel("Counts/"+str(self.exposure)+"s")
        if ylim:
            plt.ylim(-10,ylim)
        plt.title("Frame " + str(frame+1) + " out of " + str(self.numFrames))
        plt.show()

    def plotFramesColor(self, units = None, maxs = None, ylim = None):
        xaxis, xlab, _ = self.unitConversion(self.wavelengths, units = units)
        for i in range(self.numFrames):
            plt.plot(xaxis, self.data[i][0][0],color = (0, i/self.numFrames, 1-i/self.numFrames))
        #plt.xlim([2, 2.1])
        plt.xlabel(xlab)
        plt.ylabel("Counts/"+str(self.exposure)+"s")
        if ylim:
            plt.ylim(ylim)
        plt.title("Plot for Multiple Frames")
        plt.show()

    # saves the data to a csv file. If units is specified then saves it in eV
    def savetoCSV(self, filename, units = None, frame = 0):
        xaxis, xlab, _ = self.unitConversion(self.wavelengths, units = units)
        np.savetxt(filename[:-3]+"csv", np.transpose([xaxis, self.data[frame][0][0]]), delimiter = ',')

    # converts all spe files in a folder to csv
    def convert_spe_to_csv(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".spe"):
                    spe_path = os.path.join(root, filename)
                    csv_path = os.path.splitext(spe_path)[0] + ".csv"
                    spe_data = speFile3(spe_path)
                    num_frames = spe_data.numFrames
                    wavelengths = spe_data.wavelengths
                    data = spe_data.data
                    
                    df = pd.DataFrame(data[0][0][0], columns=["Frame 1"])
                    for i in range(1, num_frames):
                        df[f"Frame {i+1}"] = data[i][0][0]
                    df.insert(0, "Wavelength", wavelengths)
                    df.to_csv(csv_path, index=False)

    # unit conversions for wavelengths to specified units
    @staticmethod
    def unitConversion(xdata, units = None):
        if units == 'eV':
            xdat = h*c/ (xdata * 1e-9) / eV
            xlab = "eV"
            p0 = [100, 2, 0.1] # fit params
        elif units == 'meV':
            xdat = h*c/ (xdata * 1e-9) / eV * 1e3
            xlab = "meV"
            p0 = [100, 2, 0.1] # fit params
        else:
            xdat = xdata
            xlab = 'nm'
            p0 = [100, 620, 30] # fit params
        return xdat, xlab, p0


    # plots with a single gaussian fit
    def gaussFit(self, frame = 0, units = None):
        # Let's create a function to model and create data
        xdata, xlab, p0 = self.unitConversion(self.wavelengths, units = units)
        ydata = self.data[frame][0][0]
        def func(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))
        param, _ = curve_fit(func, xdata, ydata, p0 = p0)
  
        fit_y = func(xdata, param[0], param[1], param[2])
        plt.plot(xdata, ydata, label='Data')
        plt.plot(xdata, fit_y, '-', label='FWHM = ' + "{:.4f}".format(2.355 * param[2]) + xlab)
        plt.axvline(x = param[1], color = 'b', label = 'Center = ' + "{:.4f}".format(param[1]) + xlab)
        plt.xlabel(xlab)
        plt.ylabel("Counts/"+str(self.exposure)+"s")
        plt.title("Frame " + str(frame+1) + " out of " + str(self.numFrames))
        plt.legend(loc = 'best')
        print("Amplitude is: " + str(param[0]))
        plt.show()

    # function to sum a certain amount of frames and plot the spectra in meV
    def sumFrames(s, start, end, units = 'meV', max = None):
        sum = np.zeros(len(s.wavelengths))
        for i in range(start, end):
            sum = sum + s.data[i][0][0]
        xaxis, xlab, _ = s.unitConversion(s.wavelengths, units = units)
        plt.plot(xaxis, sum)
        #plt.xlim(540,560)
        if max:
            plt.ylim(max-1500, max)
        plt.xlabel(xlab)
        plt.ylabel("Counts/"+str(s.exposure*(end-start))+"s")
        plt.title("Sum of frames " + str(start+1) + " to " + str(end) + " out of " + str(s.numFrames))
        plt.show()
    
    # create a spectral diffusion plot based on number of frames
    # maxs is the max value for the color plot
    def specDiffusionPlot(self, units = None, maxs = None, normalized = False):
        y = []
        xaxis, xlab, _ = self.unitConversion(self.wavelengths, units = units )
        if normalized == False:
            for i in range(self.numFrames):    
                y.append(self.data[i][0][0])
        else:
            for i in range(self.numFrames): 
                maxx = max(self.data[i][0][0])   
                y.append(self.data[i][0][0]/maxx)
        times = np.arange(self.numFrames) * self.exposure
        plt.pcolormesh(xaxis,times, y, cmap="magma", vmax = maxs)
        plt.colorbar(label = "Counts/" + str(self.exposure)+'s')
        plt.xlabel(xlab)
        plt.ylabel("Time (s)")
        plt.title("Spectral Diffusion, normalized = " + str(normalized))
        plt.show()

    # interactive plot for pause play and looping through frames via animation
    def interactivePlot(self, units = None):
        # made by vasillis and chatgpt
        import ipywidgets as widgets
        from IPython.display import display

        xaxis, xlab, _ = self.unitConversion(self.wavelengths, units)

        # create figure and axes objects
        fig, ax = plt.subplots()

        # define the initial plot
        line, = ax.plot(xaxis, self.data[0][0][0])
        ax.set_title('Data set 1')
        ax.set_xlabel(xlab)
        ax.set_ylabel("Counts/"+str(self.exposure)+"s")


        # define the animation function
        def animate(frame_info):
            # update the plot with the data set for the given frame number
            frame_number = frame_info['new']
            line.set_xdata(xaxis)
            line.set_ydata(self.data[frame_number][0][0])
            ax.set_title('Data set {}'.format(frame_number + 1))
            # set other plot properties as needed
            ax.set_ylim(min(self.data[frame_number][0][0]), max(self.data[frame_number][0][0])*1.1)


        # create the animation widget
        animation_widget = widgets.Play(
            value=0,
            min=0,
            max=self.numFrames-1,  # the max value of how many datasets you have
            step=1,
            interval=1000,
            description="Play",
            disabled=False
        )

        # create the slider widget
        slider_widget = widgets.IntSlider(
            value=0,
            min=0,
            max=self.numFrames-1,  # the max value of how many datasets you have
            step=1,
            description='Frame:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )


        # define the update function for the slider
        def update_slider_value(change):
            animation_widget.value = change.new


        # link the animation and slider widgets
        widgets.jslink((animation_widget, 'value'), (slider_widget, 'value'))

        # create the previous and next buttons
        previous_button = widgets.Button(description='Previous')
        next_button = widgets.Button(description='Next')


        # define the callback functions for the buttons
        def go_previous_frame(button):
            value = max(animation_widget.value - 1, animation_widget.min)
            animation_widget.value = value


        def go_next_frame(button):
            value = min(animation_widget.value + 1, animation_widget.max)
            animation_widget.value = value

        # add the callback functions to the buttons
        previous_button.on_click(go_previous_frame)
        next_button.on_click(go_next_frame)

        # register the animate function as a callback function
        animation_widget.observe(animate, names='value')

        # display the widgets and the plot
        display(widgets.HBox([previous_button, animation_widget, next_button]))
        display(slider_widget)
        plt.show()
