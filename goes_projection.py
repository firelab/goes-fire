"""This is a class which reads and writes the goes_imager_projection container
variable in the goes 16 file."""

import pyproj
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import astropy.units as u
import astropy.coordinates as ac

class Projection(object) : 
    
    """Reads and writes the goes_imager_projection container variable in the 
      GOES-R netcdf file. Exposes the projection parameters as instance
      variables. Can produce a projection object.

   In the file, the projection information is recorded like this: 

   int32 goes_imager_projection()
      long_name: GOES-R ABI fixed grid projection
      grid_mapping_name: geostationary
      perspective_point_height: 35786023.0
      semi_major_axis: 6378137.0
      semi_minor_axis: 6356752.31414
      inverse_flattening: 298.2572221
      latitude_of_projection_origin: 0.0
      longitude_of_projection_origin: -89.5
      sweep_angle_axis: x

    The corresponding proj4 string is: 
    "+proj=geos +lon_0=-89.5 +h=35786023 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs  +sweep=x"
    """

    @classmethod
    def read_dataset(cls, ds) : 
        container = ds.variables['goes_imager_projection']
        
        p = cls()
        p.long_name                      = container.long_name
        p.grid_mapping_name              = container.grid_mapping_name
        p.perspective_point_height       = container.perspective_point_height
        p.semi_major_axis                = container.semi_major_axis
        p.semi_minor_axis                = container.semi_minor_axis
        p.inverse_flattening             = container.inverse_flattening
        p.latitude_of_projection_origin  = container.latitude_of_projection_origin
        p.longitude_of_projection_origin = container.longitude_of_projection_origin
        p.sweep_angle_axis               = container.sweep_angle_axis

        return p 

    @classmethod
    def read_ncfile(cls, fname) : 
        ds = nc.Dataset(fname)
        return cls.read_dataset(ds)

    def get_proj4_str(self) : 
        return  "+proj=geos +lon_0={} +h={} +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs +sweep={}".format(
              self.longitude_of_projection_origin,
              self.perspective_point_height,
              self.sweep_angle_axis) 

    def get_proj4_obj(self) : 
        return pyproj.Proj(self.get_proj4_str())
        
    def write_dataset(self, ds) : 
        container = ds.createVariable('goes_imager_projection', np.int32)
   
        container.long_name                      = self.long_name
        container.grid_mapping_name              = self.grid_mapping_name
        container.perspective_point_height       = self.perspective_point_height
        container.semi_major_axis                = self.semi_major_axis
        container.semi_minor_axis                = self.semi_minor_axis
        container.inverse_flattening             = self.inverse_flattening
        container.latitude_of_projection_origin  = self.latitude_of_projection_origin
        container.longitude_of_projection_origin = self.longitude_of_projection_origin
        container.sweep_angle_axis               = self.sweep_angle_axis


    def write_ncfile(self, fname) :
        ds = nc.Dataset(fname, "w") 
        self.write_dataset(ds)
     

class PixelCenters(object) : 
    """Object that contains the projection information, the 2D latitude array and 
    the 2D longitude array."""

    def __init__(self, x, y, p) : 
        self.projection = p
        p4_obj = p.get_proj4_obj()

        # prep the input data
        h = p.perspective_point_height
        x_m = x * h
        y_m = y * h 

        # construct the output arrays
        size_2d = (len(x),len(y))
        lon = ma.empty( size_2d, np.float32 )
        lat = ma.empty( size_2d, np.float32 )

        # Construct the 2D lat/lon array
        for i_y in range(len(y)) : 
            lon_row, lat_row = p4_obj(x_m, np.array([y_m[i_y]]*len(x)), inverse=True)
            lat[:,i_y] = ma.masked_values(lat_row, 1e30)
            lon[:,i_y] = ma.masked_values(lon_row, 1e30)

        self.lat = lat
        self.lon = lon

class SolarAngles(object) :
    """A class which computes solar angles for every pixel in the scene
    at a specified time. Also capable of producing masks based on 
    thresholds of zenith angle or sunglint angle"""
    def __init__(self, centers, time) : 
        self.centers = centers
        self.time    = time

        self.solar_coord = ac.get_sun(time)
        self.solar_zenith = ma.masked_all( centers.lat.shape ) 
        zenith = 90. * u.deg

        # loop over all the pixels in python, calculating the solar zenith
        # angle at each pixel center. neglect refraction and height above 
        # geoid.
        pixels = ac.EarthLocation(lat=centers.lat, lon=centers.lon, ellipsoid='GRS80')
        solar_vecs = self.solar_coord.transform_to(ac.AltAz(obstime=time,location=pixels))
        solar_zenith = zenith - solar_vecs.alt
        self.solar_zenith = ma.array(solar_zenith.to_value(u.deg), 
                                     mask=centers.lon.mask ) 

    def mask_overhead(self) : 
        return np.logical_or(self.solar_zenith.mask, self.solar_zenith.data<10)
