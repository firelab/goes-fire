"""This is a class which reads and writes the goes_imager_projection container
variable in the goes 16 file."""

import pyproj
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import astropy.units as u
import astropy.coordinates as ac
import astropy.time as at

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

    def __eq__(self, other) : 
        """Overrides the default implementation"""
        if isinstance(self, other.__class__) :
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other) : 
        """Overrides the default implementation"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))
        

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

    def __init__(self, x, y, p, lat=None, lon=None) : 
        self.projection = p
        self.x = x
        self.y = y
        self.lat = lat
        self.lon = lon

    def __eq__(self, other) : 
        """Two PixelCenter objects are equal if they have the same 
           projection parameters, and the x and y arrays are equal.
           The lat and lon arrays are derived."""

        if isinstance(self, other.__class__)  :
            return ((self.projection == other.projection) and
                   np.all(self.x == other.x) and 
                   np.all(self.y == other.y))

        return NotImplemented

    def __ne__(self, other) : 
        """Overrides the default implementation"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash( (p,x,y) )

    @property
    def lat(self) : 
        if self.__lat is None : 
            self.calc_pixel_centers()
        return self.__lat

    @lat.setter
    def lat(self,lat) : 
        self.__lat = lat

    @property
    def lon(self) : 
        if self.__lon is None : 
            self.calc_pixel_centers()
        return self.__lon

    @lon.setter
    def lon(self, lon) : 
        self.__lon = lon

    def assign_pixel_centers(self, other) : 
        """if another PixelCenter object has already calculated the lat/lon of the 
           pixel centers, it may be advantageous to just use these precalculated values.
           The projection info must be the same, as well as the x & y arrays."""
        if self == other :
            self.lat = other.lat
            self.lon = other.lon
            
    def calc_pixel_centers(self) : 
        p = self.projection
        p4_obj = p.get_proj4_obj()

        # prep the input data
        h = p.perspective_point_height
        x_m = self.x * h
        y_m = self.y * h 

        # construct the output arrays
        size_2d = (len(self.y),len(self.x))
        lon = ma.empty( size_2d, np.float32 )
        lat = ma.empty( size_2d, np.float32 )

        # Construct the 2D lat/lon array
        for i_y in range(len(self.y)) : 
            lon_row, lat_row = p4_obj(x_m, np.array([y_m[i_y]]*len(self.x)), inverse=True)
            lat[i_y,:] = ma.masked_values(lat_row, 1e30)
            lon[i_y,:] = ma.masked_values(lon_row, 1e30)

        self.lat = lat
        self.lon = lon

    @classmethod
    def read_dataset(cls, ds) : 
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        p = Projection.read_dataset(ds)

        if 'lat' in ds.variables.keys() : 
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]
            new_pc = cls(x,y,p,lat=lat,lon=lon)
        else : 
            new_pc = cls(x,y,p)
            new_pc.calc_pixel_centers()

        return new_pc

    @classmethod
    def read_ncfile(cls, fname) : 
        ds = nc.Dataset(fname)
        return cls.read_dataset(ds)

    def write_dataset(self, ds, save_centers=False) : 
        self.projection.write_dataset(ds)
        
        if 'y' not in ds.dimensions.keys() :
            ds.createDimension('y', len(self.y))
            ds.createDimension('x', len(self.x))

            y_file = ds.createVariable('y',np.float,('y',)) 
            y_file[:] = self.y

            x_file = ds.createVariable('x',np.float,('x',))
            x_file[:] = self.x

        if save_centers : 
            lat_file = ds.createVariable('lat',np.float,('y','x'))
            lat_file[:] = self.lat
            lat_file.long_name = 'latitude of pixel center'
            lat_file.units = 'deg'

            lon_file = ds.createVariable('lon',np.float,('y','x'))
            lon_file[:] = self.lon
            lon_file.long_name = 'longitude of pixel center'
            lon_file.units='deg'
     

def calc_normal_cartesian(zenith, az) : 
    """az is degrees east of north, +y is north, +x is east"""
    rad_zenith = np.radians(zenith)
    rad_az     = np.radians(az)
    z = np.cos(rad_zenith)

    r_sin_theta = np.sin(rad_zenith)
    y = r_sin_theta * np.cos(rad_az)
    x = r_sin_theta * np.sin(rad_az)

    return x,y,z

def calc_vec_angle(x1, y1, z1, x2, y2, z2) : 
    """calculates the angle between two normalized vectors. Assumes 
    vectors are normalized."""
    dot_prod = x1*x2 + y1*y2 + z1*z2
    return np.degrees(np.arccos(dot_prod))
    
class SolarAngles(object) :
    """A class which computes solar angles for every pixel in the scene
    at a specified time. Also capable of producing masks based on 
    thresholds of zenith angle or sunglint angle"""
    def __init__(self, centers, time) : 
        self.centers = centers
        self.time    = time

        self.solar_coord = ac.get_sun(time)

        # calculate the solar zenith angle at each pixel center. 
        # neglect refraction and height above geoid.
        pixels = ac.EarthLocation(lat=centers.lat, lon=centers.lon, ellipsoid='GRS80')
        solar_vecs = self.solar_coord.transform_to(ac.AltAz(obstime=time,location=pixels))

        mask = ma.getmask(centers.lon)
        self.solar_zenith = ma.array(solar_vecs.zen.to_value(u.deg), 
                                     mask=mask ) 
        self.solar_az = ma.array(solar_vecs.az.to_value(u.deg), mask=mask)

    def get_glint_az(self) : 
        glint_az = self.solar_az + 180
        glint_az = np.mod(glint_az, 360.)
        return glint_az

    def mask_overhead(self,threshold=10) : 
        return np.logical_or(self.solar_zenith.mask, self.solar_zenith.data<threshold)

    def mask_sunglint(self,sat_dir,threshold=10) : 
        glint_az = self.get_glint_az()
        glint_vec = calc_normal_cartesian(self.solar_zenith, 
                                          glint_az)
        sat_vec = calc_normal_cartesian(sat_dir.sat_zenith,
                                        sat_dir.sat_az)
        glint_angle = calc_vec_angle(*glint_vec, *sat_vec)
        return np.logical_or(self.solar_zenith.mask, 
                             glint_angle < threshold)

    def get_daytime_flag(self) : 
        if self.daytime_flag is None : 
            self.daytime_flag = np.logical_and(self.solar_zenith > 0,
                                               self.solar_zenith < 85)
        return self.daytime_flag


class LocalAngles(object) : 
    """A class which computes the local viewing angle to the satellite
    from the ground location at the center of the pixel."""
    def __init__(self, centers, sat_zenith=None,sat_az=None):
        self.centers = centers
        self.sat_lon = centers.projection.longitude_of_projection_origin
        self.sat_height = centers.projection.perspective_point_height

        self.fake_time = at.Time('2018-01-01T00:00:00Z', format='isot', scale='utc')

        self.subsatellite = ac.EarthLocation(lat=0,lon=self.sat_lon, ellipsoid='GRS80')
        self.sat_pos = ac.AltAz(alt=90*u.deg, 
                         az=0*u.deg,
                         obstime=self.fake_time,
                         distance=self.sat_height, 
                         location=self.subsatellite)

        if sat_zenith is None : 
            self.calc_viewing_angles()
        else : 
            self.sat_zenith = sat_zenith
            self.sat_az     = sat_az
      

    def calc_viewing_angles(self) : 
        """This method calculates the satellite's apparent position in the sky
        from the center of each GOES pixel. The apparent position for any given
        pixel will be constant. Hence these locations need to be calculated only
        once and may be stored in a file and re-used."""


        pixels = ac.EarthLocation(lat=self.centers.lat, 
                                  lon=self.centers.lon, 
                                  ellipsoid='GRS80')

        # for the simple case of a geostationary satellite, it may be much more 
        # computationally efficient to figure out the strictly geometric relationship
        # between the ground pixel and the satellite without requiring time.
        # however, we only need to perform this calculation once to write it to 
        # a file. After that, it is simply read in.
        sat_vecs = self.sat_pos.transform_to(ac.AltAz(obstime=self.fake_time,location=pixels))
        self.sat_zenith = ma.array(sat_vecs.zen.to_value(u.deg), 
                                     mask=self.centers.lon.mask ) 
        self.sat_az = ma.array(sat_vecs.az.to_value(u.deg), mask=self.centers.lon.mask)
    
    
    def write_dataset(self, ds, save_centers=False) : 
        """Stores the angles represented by this object into a netcdf file."""

        # creates x,y dimensions and saves coordinate variables to file
        self.centers.write_dataset(ds, save_centers)

        zenith = ds.createVariable('zenith',np.float,('y','x'), fill_value=-1000)
        zenith.long_name='satellite apparent zenith angle when viewed from ground'
        zenith.units='deg down from local zenith'
        zenith[:] = self.sat_zenith

        az = ds.createVariable('az',np.float, ('y','x'), fill_value=-1000)
        az.long_name='satellite apparent azimuth angle when viewed from ground'
        az.units='deg east of north'
        az[:] = self.sat_az

    def write_ncfile(self, fname, save_centers=False) : 
        d = nc.Dataset(fname, 'w')
        self.write_dataset(d,save_centers)
        d.close()

    @classmethod
    def read_dataset(cls,ds) : 

        p = PixelCenters.read_dataset(ds)
        zenith = ds.variables['zenith'][:]
        az     = ds.variables['az'][:]

        return cls(p,zenith,az)

    @classmethod
    def read_ncfile(cls, fname) : 
        ds = nc.Dataset(fname)
        return cls.read_dataset(ds)

