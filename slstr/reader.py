# 
# Copyright: Imperial College London.
# Licence  : GPLv3
# Created  : July 2018
#
import numpy as np

from netCDF4 import Dataset
from scipy import interpolate
from os.path import join

class Reader:
    """Sets up a primitive API for reading the netCDF files from Sentinel3, SLSTR

    # Data in example was obtained from http://data.ceda.ac.uk/neodc/sentinel3a/data/SLSTR/L1_RBT/2018/07/07/

    # Example of plotting the reflectance from the S1, S2 and S3 channels:
    from slstr.reader import Reader
    from slstr.plotter import *
    r = Reader('SLSTR/2018/07/07/S3A_SL_1_RBT____20180707T000155_20180707T000455_20180707T014632_0179_033_130_3420_SVL_O_NR_003.SEN3')
reflectance(r, 'S1', 'S2', 'S3')

    # See all possible channels
    print(r.all_channels)
   """

    
    def __init__(self, path, view='an'):
        """
        path: String that gives the directory the files are located in
        view: Two letter code for which channel and view to use
                Channels are a,b or c
                View is oblique (o) or nadir (n)
        """
        self.path = path
        self.view = view
        # S3A scaling factors
        if self.view[1] == 'n':
            self.s4_factor = 1.00
            self.s5_factor = 1.12
            self.s6_factor = 1.20
        else:
            self.s4_factor = 1.00
            self.s5_factor = 1.15
            self.s6_factor = 1.26
        self.all_channels = set(['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'F1', 'F2'])
        self.tir_channels = set(['S7', 'S8', 'S9', 'F1', 'F2'])


        self.cloud_flag = 0
        self.confidence_flag = 0
        self.pointing_flag = 0
        self.bayesian_cloud=0

        self.__first = True
        self.__channels = set()
        self.__image = {}
        self.__flag = {}
        self.__dtor = 2.0*np.pi/360.
        self.__solar_irradiance = {}
        self.__latitude = None
        self.__longitude = None
        self.__mu0 = None

        
    def radiance(self, channel):
        """Return the radiance of a given channel. checks are made to ensure
        that channel is valid. Information is cached so there is no
        performance penalty of multiple calls.
        """

        if not channel in self.__channels:
            if channel in self.tir_channels:
                self._read_tir()
                self._read_fire()
            else:
                self._read_channel(channel)

        return self.__image[channel]
        
    def reflectance(self, channel):
        """Return the radiance of a given channel. Checks are made to ensure
        that channel is valid. Information is cached so there is no
        performance penalty of multiple calls.
        """

        if not channel+'r' in self.__channels:
            self._read_reflectance(channel)

        return self.__image[channel+'r']

    def flag(self, type, offset):
        """Return an array that has either 1 or 0 depending on if the given bit is set.
 
        type   : A string that can take the values cloud, confidence, pointing and bayesian. 
        offset : An integer (starting at zero) for which bit to red.
        """
        if len(self.__flag)==0:
            self._read_flags()
            
        mask = 1 << offset
        return (self.__flag[type] & mask) >> offset

    def latitude(self):
        """Return the latitude of the points on the grid"""
        if type(self.__latitude) == type(None):
            self._fill_coords()
        return self.__latitude

    def longitude(self):
        """Return the longitude of the points on the grid"""
        if type(self.__longitude) == type(None):
            self._fill_coords()
        return self.__longitude

    def read_tir_gains(self):
        valid_tir_views = ['in', 'io']
        if self.view in valid_tir_views:
            for chan in self.tir_channels:
                qual = Dataset(join(self.path, chan + '_quality_' + self.view + '.nc'))
                setattr(self, chan.lower() + '_gain', qual.variables[chan + '_cal_gain_' + self.view][:])
                setattr(self, chan.lower() + '_offset', qual.variables[chan + '_cal_gain_' + self.view][:])
                setattr(self, chan.lower() + '_nedt_bb1', qual.variables[chan + '_dT_BB1_' + self.view][:])
                setattr(self, chan.lower() + '_nedt_bb2', qual.variables[chan + '_dT_BB2_' + self.view][:])
                setattr(self, chan.lower() + '_det_temp', qual.variables[chan + '_T_detector_' + self.view][:])
                setattr(self, chan.lower() + '_bb1_temp', qual.variables[chan + '_T_BB1_' + self.view][:])
                setattr(self, chan.lower() + '_bb2_temp', qual.variables[chan + '_T_BB2_' + self.view][:])
    
    def _read_channel(self, channel):
        """Read in a given channel from file.
        """

        if channel not in self.all_channels:
            raise Exception('The channel %s is not in the allowed list %s' % (channel, self.all_channels))
        
        valid_views = set()
        if channel in {'S1', 'S2', 'S3'}: # Visible
            valid_views = {'an', 'ao'}
        elif channel in {'S4', 'S5', 'S6'}: #IR
            valid_views = ['an', 'bn', 'ao', 'bo', 'cn', 'co']

        if self.view not in valid_views:
            raise Exception('Current view is %s but channel %s can only be read from the views %s' %
                            (self.view, channel, valid_views))

        rad = Dataset(join(self.path,channel+'_radiance_'+self.view+'.nc'))
        self.__image[channel] = rad.variables[channel+'_radiance_'+self.view][:]
        if self.__first:
            self._read_geometry(rad)

        self.__channels.add(channel)
        self.__first = False

    def _read_reflectance(self, channel):
        """Read in a given channel and convert the input to a reflectance"""

        self._read_solar_irradiance(channel)
        self.__image[channel+'r'] = self.radiance(channel) / (self.__solar_irradiance[channel] * self.__mu0) * np.pi
        self.__channels.add(channel+'r')


    def _read_tir(self):

        valid_views = ['in', 'io']

        if self.view not in valid_views:
            raise Exception('Current view is %s but TIR can only be read from the views %s' %
                            (self.view, valid_views))

        s7_bt = Dataset(self.path + 'S7_BT_' + self.view + '.nc')
        s7_bt.set_auto_scale(True)
        self.__image['S7'] = s7_bt.variables['S7_BT_' + self.view][:]

        s8_bt = Dataset(self.path + 'S8_BT_' + self.view + '.nc')
        s8_bt.set_auto_scale(True)
        self.__image['S8'] = s8_bt.variables['S8_BT_' + self.view][:]

        s9_bt = Dataset(self.path + 'S9_BT_' + self.view + '.nc')
        s9_bt.set_auto_scale(True)
        self.__image['S9'] = s9_bt.variables['S9_BT_' + self.view][:] 

        self.__channels.update({'S7', 'S8', 'S9'})

    def _read_fire(self):
        """ Read the fire channels"""

        valid_views = ['in', 'io']

        if self.view not in valid_views:
            raise Exception('Current view is %s but fire channels can only be read from the views %s' %
                            (self.view, valid_views))

        f1_bt = Dataset(self.path + 'F1_BT_' + self.view + '.nc')
        f1_bt.set_auto_scale(True)
        self.__image['F1'] = f1_bt.variables['F1_BT_' + self.view][:] 
        f2_bt = Dataset(self.path + 'F2_BT_' + self.view + '.nc')
        f2_bt.set_auto_scale(True)
        self.__image['F2'] = f2_bt.variables['F2_BT_' + self.view][:] 
        self.__channels.update({'F1', 'F2'})

    def _read_geometry(self, cdata):

        geometry = Dataset(join(self.path,'geometry_tn.nc'))
        
        solar_path    = np.where(np.isfinite(geometry.variables['solar_path_tn'][:]),
                                    geometry.variables['solar_path_tn'][:], 0.0)

        solar_azimuth = np.where(np.isfinite(geometry.variables['solar_azimuth_tn'][:]),
                                    geometry.variables['solar_azimuth_tn'][:], 0.0)
        solar_zenith  = np.where(np.isfinite(geometry.variables['solar_zenith_tn'][:]),
                                    geometry.variables['solar_zenith_tn'][:], 0.0)
        sat_path      = np.where(np.isfinite(geometry.variables['sat_path_tn'][:]),
                                    geometry.variables['sat_path_tn'][:], 0.0)
        sat_zenith    = np.where(np.isfinite(geometry.variables['sat_zenith_tn'][:]),
                                    geometry.variables['sat_zenith_tn'][:], 0.0)
        sat_azimuth   = np.where(np.isfinite(geometry.variables['sat_azimuth_tn'][:]),
                                    geometry.variables['sat_azimuth_tn'][:], 0.0)

        # Interpolate from tie point grid
        if cdata.start_offset == geometry.start_offset:
            start_offset = 0.0
        else:
            start_offset = cdata.start_offset * float(cdata.resolution.split()[2]) / float(geometry.resolution.split()[2]) - geometry.start_offset

        x = (np.array(range(cdata.dimensions['columns'].size)) - cdata.track_offset) * float(cdata.resolution.split()[1]) / float(geometry.resolution.split()[1]) + geometry.track_offset
        y = np.array(range(cdata.dimensions['rows'].size)) * float(cdata.resolution.split()[2]) / float(geometry.resolution.split()[2]) + start_offset

        geometry_rows = np.array(range(geometry.dimensions['rows'].size))
        geometry_cols = np.array(range(geometry.dimensions['columns'].size))

        f_solp = interpolate.RectBivariateSpline(geometry_rows, geometry_cols, solar_path)
        self.solar_path = f_solp(y, x)
 
        f_solz = interpolate.RectBivariateSpline(geometry_rows, geometry_cols, solar_zenith)
        self.solar_zenith = f_solz(y, x)
        self.__mu0 = np.where(self.solar_zenith < 90, np.cos(self.__dtor * self.solar_zenith), 1.0)
 
        f_sola = interpolate.RectBivariateSpline(geometry_rows, geometry_cols, solar_azimuth)
        self.solar_azimuth = f_sola(y, x)

        f_satp = interpolate.RectBivariateSpline(geometry_rows, geometry_cols, sat_path)
        self.sat_path = f_satp(y, x)

        f_satz = interpolate.RectBivariateSpline(geometry_rows, geometry_cols, sat_zenith)
        self.sat_zenith = f_satz(y, x)

        f_sata = interpolate.RectBivariateSpline(geometry_rows, geometry_cols, sat_azimuth)
        self.sat_azimuth = f_sata(y, x)


    def _read_solar_irradiance(self, channel):
        """Read the solar irradiance as required to obtain reflectance."""
        
        # Solar irradiance values are given for each detector separately, but all have same value, so just take first one for each channel here.
        self.__solar_irradiance[channel] = Dataset(join(self.path,channel+'_quality_'+self.view+'.nc')).variables[channel+'_solar_irradiance_'+self.view][0]
        
    def _read_flags(self):
        """Reads the L1 cloud product variables""" 
        flags                = Dataset(join(self.path,'flags_' + self.view + '.nc'))
        self.__flag['cloud']  = flags.variables['cloud_'+self.view][:]  # Basic cloud flag. Contains a set of binary decisions
        self.__flag['confidence'] = flags.variables['confidence_'+self.view][:] 
        self.__flag['pointing'] = flags.variables['pointing_'+self.view][:]
        self.__flag['bayesian']  = flags.variables['bayes_'+self.view][:] # More fancy.

    def _fill_coords(self):
        """Reads the longitude and latitude information"""
 
        geodetic = Dataset(join(self.path,'geodetic_' + self.view + '.nc'))
        self.__latitude  = geodetic['latitude_' + self.view][:]
        self.__longitude = geodetic['longitude_' + self.view][:]
