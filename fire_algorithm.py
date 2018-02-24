"""A completely fake and rushed implementation of 
   the GOES-R fire algorithm."""
import numpy as np
import numpy.ma as ma

def print_mask_summary(mask) : 
    totals = {} 
    for k,v in FireMask.MASK_CODES.items() : 
        print("{} : {}".format(k, np.count_nonzero(mask == v)))
    
class FireMask (object) : 

    MASK_CODES = {
        "SPACE"         : 40,
        "SOLAR_ZEN"     : 50,
        "SUNGLINT"      : 60,
        "FIRE_FREE"     : 100,
        "MISSING_CH7"   : 120,
        "MISSING_CH14"  : 121,
        "SAT_CH7"       : 123,
        "SAT_CH14"      : 124,
        "NEG_REFL"      : 125,
        "LOW_CH7"       : 126,
        "LOW_CH14"      : 127,
        "BAD_ECOSYSTEM" : 150,
        "SEA_WATER"     : 151,
        "COASTLINE"     : 152,
        "INLAND_WATER"  : 153,
        "CLOUD_CH14"    : 200,
        "CLOUD_BT_DIFF" : 205,
        "CLOUD"         : 210,
        "CLOUD_CH2"     : 215,
        "CLOUD_CH15"    : 220,
        "CLOUD_WARM"    : 225,
        "CLOUD_COLD"    : 230
    } 


    def __init__(self, lva, cmi_scene, goes_aux) : 
        self.lva = lva
        self.cmi_scene = cmi_scene
        self.goes_aux  = goes_aux
        self.refl = None
        self.mask = None
        self.minfire = None


    def get_ch(self, ch) : 
        return self.cmi_scene.channels[ch]

    def get_aux(self, ch) : 
        return self.goes_aux[ch-1]
    
    def calc_refl_product(self) : 
        """Calculate the "reflectivity product", which is defined
           as the radiance difference of band 7 - band 14 in band 7 
           space. (GOES-R Fire ATBD section 3.4.2.2)"""
        aux7 = self.get_aux(7)
        bt7  = self.get_ch(7)
        bt14 = self.get_ch(14)
        self.refl = aux7.calc_L_from_bt(bt7.data[:]) - aux7.calc_L_from_bt(bt14.data[:])

    def get_refl_product(self) : 
        if self.refl is None : 
            self.calc_refl_product()
        return self.refl
    
    def get_mask(self) : 
        if self.mask is None : 
            self.init_mask()
            self.test_sza()
            self.test_glint()
            self.test_bad_pixels()
            self.test_landcover()
            self.test_minimum_refl()
            self.test_cloudy()
        
        return self.mask

    def _mask_out_pixels(self, condition, value) : 
        """ "Masks out" pixels currently considered to be good ("FIRE_FREE")
            by changing the mask code where condition is true to the 
            specified value. This method preserves the value currently in
            the mask if it is not FIRE_FREE. 

            value is assumed to be the name of one of the codes defined above.

            The mask is assumed to already have been initialized."""
        self.mask[:] = np.where(np.logical_and(self.mask == FireMask.MASK_CODES["FIRE_FREE"],
                                               condition),
                                FireMask.MASK_CODES[value], self.mask)

    def init_mask(self) : 
        """The fire mask is an 8-bit integer which contains the results
        of the various fire pixel tests. GOES-R ATBD 3.4.2.3. This method 
        initializes the mask so that space pixels have a value of 40 and 
        pixels over the earth have a value of 100"""
        bt7 = self.get_ch(7)
        shape = bt7.get_shape()
        mask = np.empty(shape, dtype=np.uint8)

        mask[:] = np.where(ma.getmaskarray(bt7.data[:]), FireMask.MASK_CODES["SPACE"], 
                                                         FireMask.MASK_CODES["FIRE_FREE"])
        self.mask = mask

    def test_sza(self) : 
        """Tests that solar zenith angle is less than 80 deg. ATBD 3.4.2.3"""
        if self.mask is None:  
            self.init_mask()
        sva = self.cmi_scene.geo.get_solar_angles()
        self._mask_out_pixels(sva.solar_zenith > 80, "SOLAR_ZEN")

    def test_glint(self) : 
        """Tests that sza is greater than 10deg and the glint angle is 
           greater than 10 deg. ATBD 3.4.2.3"""
        sva = self.cmi_scene.geo.get_solar_angles()
        self._mask_out_pixels( np.logical_or(sva.mask_overhead(10),
                                           sva.mask_sunglint(self.lva,10)), "SUNGLINT")


    def get_min_fire_activity_flag(self) : 
        """Minimum threshold for fire activity, based on difference of ch 7 and ch 14 
           brightness temperatures. ATBD 3.4.2.3, pg 20-21"""
        if self.minfire is None : 
            bt7 = self.get_ch(7)
            bt14 = self.get_ch(14)
            self.minfire = (bt7.data[:] - bt14.data[:]) > 2
        return self.minfire
    

    def test_bad_pixels(self) : 
        """Checks for bad pixels in Ch7 and Ch 14 input data, as per ATBD 3.4.2.3"""

        # check channel 7
        bt = self.get_ch(7)
        bad = bt.quality.get_pixels_from_condition('NO_VALUE')
        self._mask_out_pixels(bad, 'MISSING_CH7')
        bad = bt.quality.get_pixels_from_condition('OUT_OF_RANGE')
        self._mask_out_pixels(bad, 'SAT_CH7')

        # check channel 14
        bt = self.get_ch(14)
        bad = bt.quality.get_pixels_from_condition('NO_VALUE')
        self._mask_out_pixels(bad, 'MISSING_CH14')
        bad = bt.quality.get_pixels_from_condition('OUT_OF_RANGE')
        self._mask_out_pixels(bad, 'SAT_CH14')
        
        

    def test_landcover(self) :
        """Checks for various invalid landcover types, as per ATBD 3.4.2.3"""
        pass

    def test_minimum_refl(self) : 
        """Checks the reflectance product is greater than zero"""
        refl = self.get_refl_product()
        self._mask_out_pixels(refl<0, 'NEG_REFL')

    def test_cloudy(self) : 
        """Applies various tests to flag clouds. ATBD 3.4.2.3. Uses thresholds
           defined for "MODIS Proxy" data."""

        bt7 = self.get_ch(7)
        bt14 = self.get_ch(14)
        
        self._mask_out_pixels(bt14.data[:] < 270, 'CLOUD_CH14')
        diff = (bt7.data[:] - bt14.data[:])
        self._mask_out_pixels(diff < -4, 'CLOUD_BT_DIFF')

        test_diff = diff > 20
        test_abs = np.logical_or( (bt7.data[:] < 285), (bt14.data[:] < 280))
        self._mask_out_pixels(np.logical_and(test_diff,test_abs), 'CLOUD')
        del test_diff
        del test_abs

        # NOTE: The tests described for channel two in ATBD 3.4.2.3 must be 
        # wrong, or at least ill-considered. The problem is in the description of
        # the criteria for a daytime pixel. I am simplifying this to sza < 70
        try: 
            refl2 = self.get_ch(2)
            sva = self.cmi_scene.geo.get_solar_angles()
            daytime = sva.calc_daytime_flag(70)
            self._mask_out_pixels(np.logical_and(daytime,refl2.data[:]>0.28), 'CLOUD_CH2')
            del daytime
        except KeyError : 
            pass

        try : 
            bt15 = self.get_ch(15)
            self._mask_out_pixels(bt15.data[:] < 265, 'CLOUD_CH15')

            test_bt14 = bt14.data[:] < 270
            tdiff = bt14.data[:] - bt15.data[:]
            self._mask_out_pixels(np.logical_and(test_bt14, tdiff < 4), 'CLOUD_WARM')
            self._mask_out_pixels(np.logical_and(test_bt14, tdiff > 60), 'CLOUD_COLD')
        except KeyError : 
            pass
     
