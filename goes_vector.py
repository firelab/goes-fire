"""
Contains the machinations necessary to take fire "ground truth" in
the form of NOAA hazard mapping system fire and extract windows
from the GOES-R CMI scenes. Also will take an approximately equal
number of windows around non-fire points. These test vectors comprise a 
training data set which can then be stored in a netcdf file (and 
subsequently read back in). 
"""

import goes_data as gd
import goes_projection as gp
import numpy as np
import netCDF4 as nc
import astropy.time as at
import astropy.units as u
import mythreads as mt
import hms


def scene_worker() : 
    """worker process to load and process a scene"""
    while True : 
        f = work_mgr.work.get()
        if f is None : 
            break 

        idx = f[0]
        args = f[1:]
        scene_pts = TrainingVector.from_scene(*args)
        work_mgr.product.put( (idx, scene_pts) )

def scene_collector(result) : 
    """prints a little status message so we know things are
       progressing"""
    print(result[0])
    work_mgr.collected_products = work_mgr.collected_products + result[1]

work_mgr = None

class TrainingVector (object) : 
    """A single training vector is the combination of data from the 
    same window in four bands of CMI data plus a day/night flag. The
    four bands are the ones used in the official WF-ABBA algorithm: 
    2, 7, 14, 15. 

    The "truth value" (whether or not the data represents a fire)
    is stored separately.
    """

    CHANNELS = (2, 7, 14, 15)
    j2000 = at.Time('2000-01-01T12:00:00Z', format='isot', scale='utc')

    def __init__(self, vector, output, center, timestamp) : 
        self.vector = vector
        self.output = output
        self.center = center
        self.timestamp = timestamp

    @property
    def size(self) : 
        return self.vector.size

    @classmethod
    def from_window(cls, scene, center, output, edge=7) : 
        windows = [gd.BandWindow(scene.channels[b].data,center,edge) 
                     for b in cls.CHANNELS]

        # lump all the channel data into one long vector.
        vector = None
        for w in windows : 
            if vector is None : 
                vector = w.data_vector
            else : 
                vector = np.concatenate( (vector, w.data_vector) )

        sva = scene.geo.get_solar_angles()
        daynight =sva.calc_daytime_flag()[center]  
        vector = np.concatenate( (vector, np.array( (daynight, )) ))

        t = scene.geo.t

        return cls( vector, output, center, t ) 
        
    @classmethod
    def from_scene(cls, fname, scene_obs) : 
        scene = gd.CMIScene.read_ncfile(fname, cls.CHANNELS)
        scene_pts = [ ]
        for i in scene_obs.index : 
            scene_pts.append(cls.from_window(scene, i, True))
        return scene_pts
        

    @classmethod
    def from_hms(cls, cmi_dir, hms_dir,lva_file) : 
        hms_inv = hms.HMSInventory(hms_dir)
        cmi_inv = gd.ABISceneInventory(cmi_dir)
        lva     = gp.LocalAngles.read_ncfile(lva_file)
        
        global work_mgr
        work_mgr = mt.ThreadManager(scene_worker, collector=scene_collector)
        work_mgr.start()
        common_years = set(hms_inv.inventory.index.levels[0]).intersection(
                        set(cmi_inv.inventory.index.levels[0]))
        for y in common_years : 
            cur_hms = hms_inv.inventory.loc[y]
            cur_cmi = cmi_inv.inventory.loc[y]
            common_days = set(cur_hms.index.values).intersection(
                           set(cur_cmi.index.levels[0]))
            for d in common_days : 
                hms_obj = hms.HMS.read_noaa(cur_hms.loc[d])

                # mask observations in scene
                hms_obj.find_scene_pixels(lva)

                # round observation times to five minutes
                hms_obj.round_obs_times()
                
                # create a properly indexed dataframe
                hms_obj = hms_obj.summarize()

                # load up the work queue
                for t in hms_obj.index.levels[0] : 
                    work_mgr.work.put( ((y,d,t), cur_cmi.loc[d,t]['Filename'], hms_obj.loc[t]) )

        # clear out the work queue
        work_mgr.collect()
        pts = work_mgr.collected_products
        work_mgr.kill()
        work_mgr = None

        return pts
            

    @classmethod
    def read_dataset(cls, ds) : 
        """Returns an array of TrainingVectors from the supplied 
        netcdf dataset, already opened."""
        vectors = ds.variables['vectors'][:]
        outputs = ds.variables['outputs'][:]

        # construct an array of center tuples
        ctr_x   = ds.variables['center_x'][:]
        ctr_y   = ds.variables['center_y'][:]
        centers = [ (y,x) for x,y in zip(ctr_x,ctr_y)]
        
        times = cls.j2000 + ds.variables['t'][:] * u.s

        return [cls(v,o,c,t) for v,o,c,t in zip(vectors,outputs,centers,times)]

    @classmethod
    def read_ncfile(cls, fname) : 
        ds = nc.Dataset(fname) 
        return cls.read_dataset(ds)

    @classmethod
    def write_dataset(cls, ds, training_data) : 
        """writes an array of training data to parallel arrays in a 
        netcdf file"""

        ds.createDimension('num_vectors', len(training_data))
        ds.createDimension('vec_size', training_data[0].size)

        vectors = ds.createVariable('vectors', np.float, ('num_vectors','vec_size'))
        outputs = ds.createVariable('outputs', np.uint8, ('num_vectors',))  
        ctr_x   = ds.createVariable('center_x', np.int,  ('num_vectors',))
        ctr_y   = ds.createVariable('center_y', np.int,  ('num_vectors',))
        t       = ds.createVariable('t', np.float, ('num_vectors',))

        t.long_name = 'time of ground truth observation'
        t.units = "seconds since J2000 epoch" 

        ctr_x.units = 'x pixel coordinate'
        ctr_y.units = 'y pixel coordinate'

        outputs.long_name = 'fire or no-fire'
        outputs.units = 'boolean flag. true indicates presence of fire'

        vectors.long_name = 'training data'
    
        vectors[:] = [ td.vector for td in training_data ]
        outputs[:] = [ td.output for td in training_data ]
        ctr_x[:]   = [ td.center[1] for td in training_data] 
        ctr_y[:]   = [ td.center[0] for td in training_data]
       
        times = at.Time( [td.timestamp for td in training_data] )
        t[:] = (times - cls.j2000).sec


    @classmethod
    def write_ncfile(cls, fname, training_data) : 
        ds = nc.Dataset(fname, 'w')
        cls.write_dataset(ds,training_data)
        ds.close()      
