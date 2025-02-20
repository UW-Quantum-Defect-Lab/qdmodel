import warnings
from typing import Dict

import numpy as np
import os
import TimeTagger


tt_filename = './2023_08_28/background/collar0/SURFACE/timetags2_2023-08-30_144612.ttbin'
filereader = TimeTagger.FileReader(tt_filename)

config: Dict = filereader.getConfiguration()
for key in config.keys():
    print(f'{key}:', config[key])

channel_resolutions = {inpt['channel'][0]: inpt['resolution rms'] for inpt in config['inputs']}
current_time = config['current time']


buffer_size = 25000000
counter = 0
while filereader.hasData():
    counter += 1
    total_size = 0
    timetags = []
    channels = []

    data = filereader.getData(buffer_size)
    event_type = np.array(data.getEventTypes())

    tt = list(np.array(data.getTimestamps())[event_type == 0])
    ch = list(np.abs(data.getChannels())[event_type == 0])

    timetags += tt
    channels += ch
    total_size += data.size

    print(len(timetags))
    print(len(channels))
    print(total_size)

    channels_collected = np.unique(channels)

    # tt_resolution = channel_resolutions[1]
    # for ch in channels_collected:
    #     if channel_resolutions[ch] != tt_resolution:
    #         warnings.warn('There are timetags from channels with different resolution')

    header = ['Time Tags',
              current_time,
              f'Unit Time Tag: {1.0}ps',
              f'Unit Channel: {1}',
              'Time Tag ; Channel']

    fmt_header = '\n'.join(header)

    txt_filename = ''.join(os.path.splitext(tt_filename)[:-1]) + f'_{counter}.txt'
    data_array = np.array([timetags, channels]).transpose()
    print(txt_filename)
    np.savetxt(txt_filename, data_array, delimiter=' ; ', fmt=('%20.0f', '%20.0f'), header=fmt_header)

