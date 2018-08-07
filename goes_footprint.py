import goes_data as gd
import goes_projection as gp
import numpy as np
import numpy.ma as ma
from shapely.geometry import Polygon, mapping
import fiona
from fiona.crs import from_epsg

class PixelFootprint (object) : 
    """Given a PixelCenters object, this class computes the footprint of each pixel 
       on the ground in geographic coordinates"""
    def __init__(self, centers) : 
        self.centers = centers
        self._ul = None
        self._ur  = None
        self._ll = None
        self._lr  = None
        self._x_corner = None
        self._y_corner = None
        self._footprint = None

    @property
    def x_corner(self) : 
        if self._x_corner is None : 
            self._x_corner = np.zeros( (len(self.centers.x)+1,), dtype=np.float64 )
            self._x_corner[1:-1] = (self.centers.x[:-1] + self.centers.x[1:]) /2
            self._x_corner[0] = self.centers.x[0] - (self.centers.x[1] - self.centers.x[0])/2
            self._x_corner[-1] = self.centers.x[-1] + (self.centers.x[-1] - self.centers.x[-2])/2
        return self._x_corner
    
    @property
    def y_corner(self) : 
        if self._y_corner is None : 
            self._y_corner = np.zeros( (len(self.centers.y)+1,), dtype=np.float64 )
            self._y_corner[1:-1] = (self.centers.y[:-1] + self.centers.y[1:]) /2
            self._y_corner[0] = self.centers.y[0] - (self.centers.y[1] - self.centers.y[0])/2
            self._y_corner[-1] = self.centers.y[-1] + (self.centers.y[-1] - self.centers.y[-2])/2
        return self._y_corner
    
    @property
    def ul(self) : 
        """PixelCenters object containing the upper left corner of each pixel"""
        if self._ul is None : 
            self._ul = gp.PixelCenters(self.x_corner[:-1], self.y_corner[1:], self.centers.projection)
        return self._ul

    @property
    def ur(self) : 
        """PixelCenters object containing the upper right corner of each pixel"""
        if self._ur is None : 
            self._ur = gp.PixelCenters(self.x_corner[1:], self.y_corner[1:], self.centers.projection)
        return self._ur

    @property
    def lr(self) : 
        """PixelCenters object containing the lower right corner of each pixel"""
        if self._lr is None : 
            self._lr = gp.PixelCenters(self.x_corner[1:], self.y_corner[:-1], self.centers.projection)
        return self._lr

    @property
    def ll(self) : 
        """PixelCenters object containing the lower left corner of each pixel"""
        if self._ll is None : 
            self._ll = gp.PixelCenters(self.x_corner[:-1], self.y_corner[:-1], self.centers.projection)
        return self._ll

    @property
    def footprint(self) : 
        """returns a masked array where the items are either polygons containing the footprint
        of the pixel, or masked out..."""
        if self._footprint is None : 
            elements = self.ll.lon.size
            polys = [ ]
            for j in range(self.ll.lon.shape[0]) : 
                polys.append( [ Polygon([ (i[0], i[1]), (i[2],i[3]), (i[4],i[5]), (i[6],i[7]), (i[0],i[1])]) 
                                                  if i[0] is not ma.masked else None
                                                  for i in zip(self.ll.lon[j,:], 
                                                               self.ll.lat[j,:],
                                                               self.ul.lon[j,:],
                                                               self.ul.lat[j,:],
                                                               self.ur.lon[j,:],
                                                               self.ur.lat[j,:],
                                                               self.lr.lon[j,:],
                                                               self.lr.lat[j,:]) ])

            self._footprint = polys
        return self._footprint


    @classmethod
    def read_dataset(cls, ds) : 
        abi_data = gd.ABIData.read_L2_dataset(ds)
        return cls(abi_data.centers)

    @classmethod
    def read_ncfile(cls, fname) : 
        abi_data = gd.ABIData.read_L2_ncfile(fname)
        return cls(abi_data.centers)

    def write_footprint(self, fname) : 
        schema =  { 'geometry' : 'Polygon', 'properties': {'i':'int', 'j':'int'} }
        with fiona.open(fname, 'w', schema=schema, driver='ESRI Shapefile', crs=from_epsg(4326)) as f : 
            for j in range(len(self.footprint)) : 
                for i in range(len(self.footprint[0])) : 
                    f.write( { 'geometry' : mapping(self.footprint[j][i]), 
                               'properties' : {'i': i, 'j':j } } ) 
