"""Utilities to manipulate data"""
import goes_projection as gp
import numpy.ma as ma
import numpy as np
import netCDF4 as nc
import astropy.units as u
import astropy.time as at
import pandas as pd
import glob
import os.path
from datetime import datetime

class ABIData (object) : 
    """ABI Data are data collected from a geostationary platform. 
    The data are expected to be from the vantage point of one of the
    "standard" GOES slots: Goes East, Goes West, or Goes Test. 
    The data may be a full disk scene, a CONUS scene, or it may 
    be a scene of a mesoscale event of interest.
    The data may be from a channel on the instrument, or it may
    be a derived product (standard or otherwise). 

    This class is mainly concerned with geolocating pixels in a 
    scene. 
    """
    def __init__(self, x, y, proj, t, scene="CONUS", platform="G16") : 

        self.scene = scene
        self.platform = platform
        self.proj = proj 
        self.x    = x
        self.y    = y
        self.t    = t

        self.centers = None
        self._sza = None

    @property
    def centers(self) : 
        if self.__centers is None :
            self.centers = gp.PixelCenters(self.x, self.y, self.proj) 
        return self.__centers

    @centers.setter
    def centers(self, centers) : 
        """When we're assigning a "pixel centers" object, make sure the projection, 
           x and y  arrays match our scene data."""
        if centers is None : 
           self.__centers = None
        elif ((self.proj == centers.projection) and 
           np.all( self.x == centers.x ) and 
           np.all( self.y == centers.y )) : 
           self.__centers = centers

    def get_solar_angles(self) : 
        if self._sza is None : 
            pc = self.centers
            self._sza = gp.SolarAngles(pc, self.t)
        return self._sza


    @classmethod
    def read_L2_dataset(cls, ds) : 
        scene = ds.scene_id
        platform = ds.platform_ID
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        proj = gp.Projection.read_dataset(ds)
        j2000 = at.Time('2000-01-01T12:00:00Z', format='isot', scale='utc')
        t = j2000 + ds.variables['t'][:] * u.s
        return cls(x, y, proj, t, scene, platform)

    @classmethod
    def read_L2_ncfile(cls, fname) : 
        ds = nc.Dataset(fname)
        return cls.read_L2_dataset(ds)

        
        
class Feature(object) : 
    """Gridded data which represent either a measurement or a 
    derived product."""

    def __init__(self, data, id, min=None, max=None, mean=None, stdev=None, outlier_count=None) : 
        self.data = data
        self.id = id
        self.min = min
        self.max = max
        self.mean = mean
        self.stdev = stdev
        self.outlier_count = outlier_count

class DetectorBand(Feature) : 
    """A band of data associated with one of the detectors on
    an instrument. The data will have a center wavelength associated
    with the detector.  It may also have a spectral response function.
    Depending on the level of processing, the data may be 
    raw counts, calibrated radiance, top of the atmosphere reflectance,
    top of the atmosphere brightness temperature, or atmospherically
    corrected ground leaving radiance/reflectance/brightness temperature.
    Consult the ATBD for the relevant data product. 

    Data arrays are represented as netCDF4 variables."""

    def __init__(self, data, wavelength, id, min=None, max=None, mean=None, stdev=None, outlier_count=None) : 
        super(DetectorBand,self).__init__(data, id, min, max, mean, stdev, outlier_count) 
        self.wavelength = wavelength

class CMIBandQuality(object) : 
    """Contains the band quality information for the Cloud and Moisture Imagery,
    and the ability to generate masks based on data quality."""

    CODES = {
        'GOOD'                 : np.uint8(0),
        'CONDITIONALLY_USABLE' : np.uint8(1),
        'OUT_OF_RANGE'         : np.uint8(2),
        'NO_VALUE'             : np.uint8(3)
    }

    def __init__(self, ds, channel) : 
        self.data = ds.variables['DQF_C{:02d}'.format(channel)]

    def get_pixels_from_condition(self, code_text) : 
        """User specifies one of the condition codes by text, 
        function returns a boolean array of locations where 
        the pixel is flagged with that condition."""
        return (self.data[:] == CMIBandQuality.CODES[code_text])

class CMIBand (DetectorBand) : 
    """Cloud and Moisture Imagery is essentially scaled radiance. 
    Reflective bands are in units of top of the atmosphere reflectance
    multiplied by the cosine of the solar zenith angle. Emissive bands 
    are in units of top of the atmosphere effective temperature. See the 
    CMI ATBD for more information.

    This class serves mainly as a holder of NetCDF variables, so that
    the attributes or data can be examined as desired."""

    REFLECTIVE_CHANNELS = range(1,7)
    

    def __init__(self, ds, channel) : 
        self.data = ds.variables['CMI_C{:02d}'.format(channel)]
        self.ds   = ds
        self.channel = channel
        self.quality = CMIBandQuality(ds, channel)

        self.wavelength = self.ds.variables['band_wavelength_C{:02d}'.format(self.channel)]
        self.id = self.ds.variables['band_id_C{:02d}'.format(self.channel)]
        self.outliers = self.ds.variables['outlier_pixel_count_C{:02d}'.format(self.channel)]

        if channel in CMIBand.REFLECTIVE_CHANNELS : 
            self._read_reflective()
        else: 
            self._read_emissive()
    
    def _read_emissive(self) : 
        self.min = self.ds.variables['min_brightness_temperature_C{:02d}'.format(self.channel)]
        self.max = self.ds.variables['max_brightness_temperature_C{:02d}'.format(self.channel)]
        self.mean = self.ds.variables['mean_brightness_temperature_C{:02d}'.format(self.channel)]
        self.stdev = self.ds.variables['std_dev_brightness_temperature_C{:02d}'.format(self.channel)]

    def _read_reflective(self) : 
        self.min = self.ds.variables['min_reflectance_factor_C{:02d}'.format(self.channel)]
        self.max = self.ds.variables['max_reflectance_factor_C{:02d}'.format(self.channel)]
        self.mean = self.ds.variables['mean_reflectance_factor_C{:02d}'.format(self.channel)]
        self.stdev = self.ds.variables['std_dev_reflectance_factor_C{:02d}'.format(self.channel)]

    def get_shape(self) : 
        return self.data.shape

class ABIScene(object) : 
    """Combines the geospatial information for the scene with one or more bands of data.
    The bands are stored in a dictionary which is keyed by the channel number. One of the 
    implicit constraints is that the data arrays have the same resolution."""
    def __init__(self, geo, channels) : 
        self.geo = geo
        self.channels = channels

class CMIScene(ABIScene) : 

    @classmethod
    def read_dataset(cls, ds, channels, pc=None) : 
        geo = ABIData.read_L2_dataset(ds)

        if pc is not None : 
            geo.centers = pc

        channel_dict = {}
        for ch in channels : 
            channel_dict[ch] = CMIBand(ds, ch)

        return cls(geo, channel_dict)

    @classmethod
    def read_ncfile(cls, fname, channels, pc=None) : 
        ds = nc.Dataset(fname)
        return cls.read_dataset(ds, channels, pc)

class FireScene (ABIScene) : 
    """A GOES-R fire product primarily consists of mask, temperature,
       area, and radiative power arrays."""
    pass

class SceneDate(object) : 
    """can parse or format a timestamp found in various locations
       in the filename."""
    FORMAT_STR = '%Y%j%H%M%S'

    def __init__(self) : 
        self.__index = None

    @classmethod
    def parse(cls, string) : 
        """given a string, which is expected to be the entire field
        in the filename which encodes one of the three important times
        (observation start, observation end, processing), this method 
        produces a scenedate object."""

        scene = cls()
        time = datetime.strptime(string[1:-1], cls.FORMAT_STR)
        scene.code = string[0]
        tenths = int(string[-1])
        microsecond = 100000 * tenths
        scene.time = datetime(time.year, time.month, time.day,
                             time.hour, time.minute, time.second,
                             microsecond)

        return scene

    @classmethod
    def from_obj(cls, obj, code=None) : 
        """creates a scenedate object from a datetime and a type code"""
        scene = cls()
        scene.time = obj
        scene.code = code
        return scene

    def format(self) : 
        """returns a string which could be placed in the filename"""
        # integer division should result in single digit 0-9
        tenths = self.time.microsecond / 100000
        code = 'u'
        if self.code is not None : 
            code = self.code
        return code + self.time.strftime(self.FORMAT_STR) + str(tenths)

    @property
    def clocktime(self) : 
        """returns an integer representing the 24 hour clock time
        (hours and minutes)."""
        return self.time.hour*100 + self.time.minute

    @property
    def index(self) : 
        """Returns a tuple of fields which can be used as an index:
        (year, doy, 24-time) where each item is an integer."""
        if self.__index is None : 
            rt = self.rounded_clocktime()
            doy = int(self.time.strftime('%j'))
            self.__index = (self.time.year, doy, rt)
        return self.__index
        

    def rounded_clocktime(self, nearest=5) : 
        """returns an integer representing the 24 hour clock time
        rounded to "nearest" minutes"""
        mins = int(nearest * np.round(self.time.minute/float(nearest)))
        hours = self.time.hour
        if mins >= 60 : 
            mins =0
            hours = hours + 1
        return hours*100 + mins
        
class ABISceneInventory (object) : 
    """Given a directory, scans for all the files matching a particular
    pattern. The raw list of filenames is presented as the files attribute,
    and a dataframe of filename, start, stop and end times is presented
    as the inventory attribute."""
    def __init__(self, dirname) : 
        self.dirname = dirname
        self.files = glob.glob(os.path.join(dirname,'OR_ABI*.nc'))
        self._get_times()

    def _get_times(self, nearest = None) : 
        start_times = [None] * len(self.files)
        end_times = [None] * len(self.files)
        proc_times = [None] * len(self.files)
        indices = [None] * len(self.files)
        for i in range(len(self.files)) : 
            fields = self.files[i].split('_') 

            start_times[i] = SceneDate.parse(fields[3])
            end_times[i]   = SceneDate.parse(fields[4])

            # last field has the ".nc" at the end which we need to
            # get rid of.
            pfields = fields[5].split('.')
            proc_times[i]  = SceneDate.parse(pfields[0])

            # getting the indices
            indices[i] = start_times[i].index

        i = pd.MultiIndex.from_tuples(indices, names=['year','doy','time'])
        self.inventory = pd.DataFrame({"Filename" : self.files,
                                       "start"    : start_times,
                                       "end"      : end_times,
                                       "proc"     : proc_times}, index=i)

