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
import numpy.random as nr
import netCDF4 as nc
import astropy.time as at
import astropy.units as u
import mythreads as mt
import pandas as pd
import imageio
import hms


# 
# Code which supports multithreaded operation
#
def scene_worker() : 
    """worker process to load and process a scene"""
    while True : 
        f = work_mgr.work.get()
        if f is None : 
            break 

        idx = f[0]
        args = f[1:]
        scene_pts = GOESVector.from_scene(*args)
        work_mgr.product.put( (idx, scene_pts) )

def scene_collector(result) : 
    """prints a little status message so we know things are
       progressing"""
    print(result[0])
    work_mgr.collected_products = work_mgr.collected_products + result[1]

work_mgr = None


class Dummy(object): 
    """I just need to have something to hang an attribute on"""
    pass

def normalize(v, limits) :
    return (v - limits[0]) / (limits[1] - limits[0])
    

class GOESVector (object) : 
    """A single goes vector is the combination of data from the 
    same window in four bands of CMI data plus a day/night flag. The
    four bands are the ones used in the official WF-ABBA algorithm: 
    2, 7, 14, 15. 

    The "truth value" (whether or not the data represents a fire)
    is stored separately.
    """

    # valid ranges for data taken from the GOES PUG L2+ vol 5,
    # table 5.1.6.4-1
    CHANNELS = (2, 7, 14, 15)
    VALID_RANGES = ( (0.,1.),          # ch 2
                     (197.31, 411.86), # ch 7
                     (96.19, 341.28),  # ch 14
                     (97.38, 341.28))  # ch 15
    BAD_LANDCOVER = (0,11,13,15)
    j2000 = at.Time('2000-01-01T12:00:00Z', format='isot', scale='utc')

    def __init__(self, vector, output, center, timestamp) : 
        self.vector = vector
        self.output = output
        self.center = center
        self.timestamp = timestamp

    @property
    def size(self) : 
        return self.vector.size

    @property
    def window_size(self):
        return int((self.size -1)/len(self.CHANNELS))

    @property
    def window_edge(self) : 
        return int(np.sqrt(self.window_size))


    @property
    def normal_vector(self) : 
        """Normalizes the vector to the valid range of 
        allowed values on a per-channel basis."""
        normal = np.array(self.vector)
        v_size = self.window_size
        for i in range(len(self.VALID_RANGES)) : 
            normal[i*v_size:(i+1)*v_size] = normalize(
                     self.vector[i*v_size:(i+1)*v_size], 
                     self.VALID_RANGES[i])
        return normal

    @property
    def windows(self) : 
        """Reconstruct the "windows" of data"""
        w_data = self.vector[:-1]
        w_edge = self.window_edge
        return w_data.reshape( (len(self.CHANNELS),w_edge,w_edge) )

    @property
    def normal_windows(self) : 
        """Reconstruct the "windows" of data"""
        w_data = self.normal_vector[:-1]
        w_edge = self.window_edge
        return w_data.reshape( (len(self.CHANNELS),w_edge,w_edge) )


    @classmethod
    def from_window(cls, scene, center, output, edge=7) : 
        """Constructs a single GOESVector from the bands
        in the given scene. The extract from each band is 
        a window centered on the pixel specified by "center",
        having a length of "edge" pixels along each side.
        Output is a flag which indicates whether there is a 
        fire in the window or not."""
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

        t = scene.geo.t_start

        return cls( vector, output, center, t, windows ) 
        
    @classmethod
    def from_scene(cls, fname, scene_obs, output) : 
        """Given the filename of a GOES CMI scene and a collection
        of hms fire observations concurrent with the scene, extracts a 
        GOESVector centered on each hms observation. This is useful for 
        collecting training data."""
        scene = gd.CMIScene.read_ncfile(fname, cls.CHANNELS)
        scene_pts = [ ]
        for i in scene_obs.index : 
            scene_pts.append(cls.from_window(scene, i, output))
        return scene_pts
        

    @classmethod
    def from_hms(cls, cmi_dir, hms_dir,lva_file) : 
        """Given a directory of CMI images, another directory
        of HMS text files, and a "local viewing angle" file for 
        the satellite slot, this code finds the common observation
        times and extracts training data around each known fire."""
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
                    work_mgr.work.put( ((y,d,t), cur_cmi.loc[d,t]['Filename'], 
                                         hms_obj.loc[t], True) )

        # clear out the work queue
        work_mgr.collect()
        pts = work_mgr.collected_products
        work_mgr.kill()
        work_mgr = None

        return pts
            
    @classmethod 
    def nonfire(cls, fire_pt_file, cmi_dir, lc_file) : 
        """Reads in the fire point training file in order to pick
        non-fire points. Eliminates any pixel containing fire in any of the 
        scenes contributing to the training set. Also eliminates "bad" 
        landcovers types: 

          0 : Water
         11 : 
         13 : 
         15 : 
        """
        # read in the input datasets
        cmi_inv = gd.ABISceneInventory(cmi_dir)
        fires   = cls.read_ncfile(fire_pt_file)
        lc      = imageio.imread(lc_file)

        # extract indices of all the fire pixels
        i_fire  = [ np.array( [f.center[0] for f in fires]),
                    np.array( [f.center[1] for f in fires]) ] 
       
        #initialize the fire mask
        mask    = np.zeros(lc.shape, dtype=np.bool) 
        mask[i_fire] = True

        # now filter out all pixels we're not going to allow
        # to burn
        for lc_code in cls.BAD_LANDCOVER :
            mask = np.where(lc == lc_code, 1, mask)
        
        # now randomly pick some non-fire pixels.
        i_candidates = np.where(mask==0)
        nonfire = nr.randint(0, len(i_candidates[0]), size=(len(fires),))
        i_nonfire = [ i_candidates[0][nonfire],
                      i_candidates[1][nonfire] ]


        # and randomly pick the scenes from which the pixels
        # are drawn 
        unique_scenes = set( [ f.timestamp.datetime for f in fires ] )
        unique_scenes = list( unique_scenes ) 
        scenes = nr.randint(0, len(unique_scenes), size=(len(fires),))
        scene_timestamps = [ unique_scenes[i] for i in scenes ] 

        # compose a dataframe representing the non-fire scenes/pixels
        centers =[ c for c in zip( i_nonfire[0], i_nonfire[1] ) ]
        fnames = [] 
        indices = []
        for i in range(len(fires)) : 
            cur_scene_date = gd.SceneDate.from_obj(scene_timestamps[i])
            indices.append(cur_scene_date.index )
    
            cur_fname = cmi_inv.inventory.loc[cur_scene_date.index]['Filename']
            if type(cur_fname) is not str : 
                cur_fname = cur_fname[0]
            fnames.append( cur_fname ) 
        i_df = pd.MultiIndex.from_tuples(indices, names=['Year', 'doy', 'time'])
        df = pd.DataFrame( {'Filename':fnames, 'Center':centers}, index=i_df)
        df = df.sort_index()
      
        # set up the workers
        global work_mgr
        work_mgr = mt.ThreadManager(scene_worker, collector=scene_collector)
        work_mgr.start()


        # load up the queue
        unique_i_df = set( [ i for i in i_df ] ) 
        sum = 0
        for y,d,t in unique_i_df : 
            cur_scene = df.loc[y,d,t]
            scene_pix = Dummy()
            scene_pix.index = cur_scene['Center'].tolist()
            sum = sum + len(scene_pix.index)
            fname = cur_scene['Filename'].iloc[0]
            work_mgr.work.put( ((y,d,t), fname, scene_pix, False) )

        print(sum)
        
        # clear out the work queue
        work_mgr.collect()
        pts = work_mgr.collected_products
        work_mgr.kill()
        work_mgr = None

        return pts                           
        
            
    @classmethod
    def read_dataset(cls, ds) : 
        """Returns an array of GOESVectors from the supplied 
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

        t.long_name = 'start time of satellite observation'
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
