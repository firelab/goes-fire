"zeniti""Utilities to manipulate data"""
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

        self._pc = None 
        self._sza = None

    def get_pixel_centers(self) : 
        if self._pc is None :
            self._pc = gp.PixelCenters(self.x, self.y, self.proj) 
        return self._pc

    def get_solar_angles(self) : 
        if self._sza is None : 
            pc = self.get_pixel_centers()
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

        
        


class DetectorBand(object) : 
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
        self.data = data
        self.wavelength = wavelength
        self.id = id
        self.min = min
        self.max = max
        self.mean = mean
        self.stdev = stdev
        self.outlier_count = outlier_count

class CMIBandQuality(object) : 
    """Contains the band quality information for the Cloud and Moisture Imagery,
    and the ability to generate masks based on data quality."""

    def __init__(self, ds, channel) : 
        self.data = ds.variables['DQF_C{:02d}'.format(channel)]

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

class ABIScene(object) : 
    """Combines the geospatial information for the scene with one or more bands of data.
    The bands are stored in a dictionary which is keyed by the channel number. One of the 
    implicit constraints is that the data arrays have the same resolution."""
    def __init__(self, geo, channels) : 
        self.geo = geo
        self.channels = channels

class CMIScene(ABIScene) : 

    @classmethod
    def read_dataset(cls, ds, channels) : 
        geo = ABIData.read_L2_dataset(ds)
        channel_dict = {}
        for ch in channels : 
            channel_dict[ch] = CMIBand(ds, ch)

        return cls(geo, channel_dict)

    @classmethod
    def read_ncfile(cls, fname, channels) : 
        ds = nc.Dataset(fname)
        return cls.read_dataset(ds, channels)
