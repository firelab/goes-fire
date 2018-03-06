import pandas as pd
import numpy as np
import os.path
import glob
import pyproj
from datetime import datetime

def round_times(times) : 
    return rt

class HMS (object) : 
    """Encapsulates a hazard mapping system text file."""
    def __init__(self, hms) : 
        """Initializes this file given the text file name. Assumes
        the file provided is in NOAA format: CSV with headers."""
        self.hms = hms


    @classmethod
    def read_noaa(cls, fname) : 
        hms = pd.read_csv(fname)

        # strip out all the extra spaces that NOAA puts in the file
        hms.columns = hms.columns.str.strip()
        hms['Satellite'] = hms['Satellite'].str.strip()
        hms['Method of Detect'] = hms['Method of Detect'].str.strip()

        return cls(hms)

    def find_scene_pixels(self, local_angles) :
        """Given a LocalAngles object representative of a particular 
        satellite viewing geometry, locate the pixel in which each 
        HMS observation occurs. Also filter out observations which are
        not in the satellite scene. Adds two extra columns to the 
        hms PANDAS object: i_x and i_y."""

        # convert all the lat/lons in the hms file to pixels in the 
        # ABI scene
        nad80 = pyproj.Proj('+proj=longlat +ellips=grs80')
        i_x, i_y = local_angles.to_image(self.hms['Lon'].values,self.hms['Lat'].values,nad80)

        # store these pixel centers as new columns 
        self.hms['i_x'] = i_x
        self.hms['i_y'] = i_y

        # eliminate all the hms observations not in the goes scene
        self.hms = self.hms[np.logical_not(i_x.mask)]

    def round_obs_times(self, nearest=5) : 
        """round the hms observation times to "nearest" minutes.
        Adds a column called "Rounded Time" to the hms PANDAS object.""" 
        rt = np.array( nearest * np.round(self.hms['Time'].values/float(nearest)), dtype=np.int)
        overshoot = np.mod(rt, 100) >= 60
        rt = np.where(overshoot, 
                      (np.floor_divide(rt,100) + 1) * 100 ,
                      rt)
        self.hms['Rounded Time'] = rt

    def summarize(self, time_col="Rounded Time") : 
        """Produces and returns a summary PANDAS object which has a count
        of observations per GOES pixel and observation time. Defaults to
        using the Rounded observation time column, but allows the caller
        to specify a different column."""
        return self.hms.groupby( [time_col, 'i_y', 'i_x'])['Satellite'].count()

class HMSDate(object) : 
    """class to transition back and forth between the date of the HMS 
    observation file and the filename"""

    def __init__(self) : 
        self.__index = None

    @classmethod
    def parse(cls, string) : 
        hms = cls()
        bn = os.path.basename(string)
        datestring = bn[3:11]
        hms.time = datetime.strptime(datestring, '%Y%m%d').date()
        return hms
        

    def format(self) : 
        return self.time.strftime('hms%Y%m%d.txt')

    @property
    def index(self) : 
        """returns a (year, doy) tuple for use as an index"""
        if self.__index is None : 
            doy = int(self.time.strftime('%j'))
            self.__index = (self.time.year, doy)

        return self.__index
        

class HMSInventory (object) : 
    """scans a directory for hazard mapping system files and 
    retains a list"""
    def __init__(self, directory) : 
        self.files = glob.glob(os.path.join(directory, 'hms*.txt'))
        hms_times = [ HMSDate.parse(d) for d in self.files ]
        i_tuple = [i.index for i in hms_times ]
        idx = pd.MultiIndex.from_tuples(i_tuple, names=['year','doy'])
        self.inventory = pd.Series(self.files, index=idx)


