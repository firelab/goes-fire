"""Utilities to manipulate data"""
import goes_projection as gp
import numpy.ma as ma
import numpy as np
import netCDF4 as nc
import astropy.units as u
import astropy.time as at

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
