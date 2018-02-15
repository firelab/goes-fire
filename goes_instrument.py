"""
Classes and utilities to manipulate parameters which largely are 
fixed by the instrument characteristics, primarily the spectral
response function of each channel. The parameters are precomputed
approximations for conversion between radiance and either 
reflectance or brightness temperature.

These parameters must be: 
  1] harvested from the L1b files (one time)
  2] stored in an ancillary file (one time)
  3] read from the ancillary file (each scene)
  4] used to convert data (each scene) 


CAUTIONARY NOTE: 
  The assumption which allows us to harvest calibration parameters from
  the L1b files and apply them to unrelated L2 files is that these 
  instrument parameters do not vary with time. The kappa0 parameter 
  actually does, as it depends on the earth-sun distance and potentially 
  the measured solar irradiance. Conversion between reflectance and 
  radiance will work correctly if the kappa0 parameter is harvested 
  from the corresponding L1b scene, or one collected at very nearly 
  the same time. Kappa0 varies by approximately 5% from perihelion to
  aphelion. 

  The emissive parameters are constant with time.
"""
import netCDF4 as nc  
import numpy as np
import glob

CAL_VARIABLES = ('kappa0', 
                 'planck_fk1', 
                 'planck_fk2', 
                 'planck_bc1',
                 'planck_bc2')

# stolen from https://gitlab.ssec.wisc.edu/rayg/goesr/blob/master/goesr/l1b.py
def _calc_bt(L: np.ndarray, fk1: float, fk2: float, bc1: float, bc2: float):
    return (fk2 / np.log((fk1 / L) + 1.0) - bc1) / bc2

def _calc_L_from_bt(T: np.ndarray, 
                    fk1: float, 
                    fk2: float, 
                    bc1: float, 
                    bc2: float):
    return fk1 / (np.expm1(fk2/ ((T * bc2) + bc1)) )

#stolen from https://gitlab.ssec.wisc.edu/rayg/goesr/blob/master/goesr/l1b.py
def _calc_bt_wrap(L: (np.ndarray, np.ma.masked_array), fk1: float, fk2: float, bc1: float, bc2: float, mask_invalid: bool=True, **etc):
    """
    convert brightness temperature bands from radiance
    ref PUG Vol4 7.1.3.1 Radiances Product : Description
    Note: marginally negative radiances can occur and will result in NaN values!
    We handle this by setting anything with radiance <=0.0 to return 0K
    :param L: raw radiance image data, preferably as masked array
    :param fk1: calibration constant
    :param fk2: calibration constant
    :param bc1: calibration constant
    :param bc2: calibration constant
    :param mask_invalid: bool, whether to include radiances <=0.0 in array mask - iff Lis masked array
    :return: BT converted radiance data, K; if input is masked_array, L<=0.0 will also be masked
    """
    T = _calc_bt(L, fk1, fk2, bc1, bc2)
    if isinstance(L, np.ma.masked_array):
        if mask_invalid:
            T = np.ma.masked_array(T, (L.data <= 0.0) | L.mask)
        else:
            T = np.ma.masked_array(T, L.mask)
    else:
        # Tmin = -bc1 / bc2   # when L<=0
        T[L <= 0.0] = 0.0  # Tmin, but for now truncate to absolute 0
    return T

#stolen from https://gitlab.ssec.wisc.edu/rayg/goesr/blob/master/goesr/l1b.py
def _calc_refl(L: np.ndarray, kappa0: float, **etc):
    """
    convert reflectance bands from radiance
    ref PUG Vol4 7.1.3.1 Radiances Product : Description
    :param L: raw radiance image data as masked array
    :param kappa0: conversion factor radiance to reflectance
    :return:
    """
    return L * kappa0



class ABI_Channel_Characteristics (object) : 

    def __init__(self, kappa0, planck_fk1, planck_fk2, planck_bc1, planck_bc2) : 
        self.kappa0 = kappa0
        self.planck_fk1 = planck_fk1
        self.planck_fk2 = planck_fk2
        self.planck_bc1 = planck_bc1
        self.planck_bc2 = planck_bc2

    @classmethod
    def read_l1b_dataset(cls, ds) : 
        vals = [ ]
        for vname in CAL_VARIABLES : 
            vals.append(float(ds.variables[vname][:]))
        return cls(*vals)

    @classmethod
    def read_l1b_ncfile(cls, fname) : 
        ds = nc.Dataset(fname) 
        return cls.read_l1b_dataset(ds)


    def to_params(self) :
        return (self.kappa0,
                self.planck_fk1,
                self.planck_fk2,
                self.planck_bc1,
                self.planck_bc2)

    def is_reflective(self) : 
        return np.isnan(self.planck_fk1)

    def is_emissive(self) : 
        return np.isnan(self.kappa0)

    def calc_refl(self, L) : 
        return _calc_refl(L, self.kappa0)

    def calc_bt(self, L) : 
        return _calc_bt_wrap(L, self.planck_fk1,
                                self.planck_fk2,
                                self.planck_bc1,
                                self.planck_bc2)

    def calc_L_from_bt(self, T) : 
        return _calc_L_from_bt(T, self.planck_fk1,
                                  self.planck_fk2,
                                  self.planck_bc1,
                                  self.planck_bc2)


def read_l1b_files(fname_pattern) : 
    """The filename pattern must expand to match the set of L1b files
    which contain data for each of the ABI channels. The code will 
    determine the channel which is referred to by looking at the 
    band_id variable"""
    channel_list = [ None ] * 16
    files = glob.glob(fname_pattern) 
    if len(files) == 16 : 
        for f in files : 
            ds = nc.Dataset(f)
            ch = int(ds.variables['band_id'][:])
            channel_list[ch-1] = ABI_Channel_Characteristics.read_l1b_dataset(ds)

    return channel_list

def read_aux_file(fname) : 
    ds = nc.Dataset(fname)
    channel_list = [] 
    for i in range(16) : 
        params = [ds.variables[vname][i] for vname in CAL_VARIABLES]
        channel_list.append( ABI_Channel_Characteristics( *params ) )
    ds.close() 
    return channel_list

def write_aux_file(fname, channel_list) : 
    ds = nc.Dataset(fname, 'w') 
    ds.createDimension('channels', 16)
    for vname in CAL_VARIABLES: 
        ds.createVariable(vname, np.float, dimensions=('channels',))

    params = [ [] for i in range(len(CAL_VARIABLES))]
    for i_ch in range(16) : 
        cur_params = channel_list[i_ch].to_params()

        for i_v in range(len(CAL_VARIABLES)) : 
            params[i_v].append( cur_params[i_v] )

    for i_v in range(len(CAL_VARIABLES)) : 
        vname = CAL_VARIABLES[i_v]
        ds.variables[vname][:] = params[i_v]

    ds.close()
