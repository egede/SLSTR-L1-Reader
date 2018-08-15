# 
# Copyright: Imperial College London.
# Licence  : GPLv3
# Created  : July 2018
#
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from scipy import interpolate
from time import time
from os.path import join

class Reader:
    """Sets up a primitive API for reading the netCDF files from Sentinel3, SLSTR
    
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


        self.rgb = [0, 0, 0]
        self.latitude = [0, 0]
        self.longitude = [0, 0]
        self.cloud_flag = 0
        self.confidence_flag = 0
        self.pointing_flag = 0
        self.bayesian_cloud=0

        self.__first = True
        self.__channels = set()
        self.__image = {}
        self.__dtor = 2.0*np.pi/360.
        self.__solar_irradiance = {}

    def radiance(self, channel):
        """Return the radiance of a given channel. checks are made to ensure
        that channel is valid. Information is cached so there is no
        performance penalty of multiple calls.
        """

        if not channel in self.__channels:
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
        
    def _read_channel(self, channel):
        """Read in a given channel from file.
        """

        if channel not in self.all_channels:
            raise Exception('The channel %s is not in the allowed list %s' % (channel, self.all_channels))
        
        if self.__first:
            self._fill_coords()
            self._read_flags()

        valid_views = set()
        if channel in {'S1', 'S2', 'S3'}: # Visible
            valid_views = {'an', 'ao'}
        elif channel in {'S4', 'S5', 'S6'}: #IR
            valid_views = ['an', 'bn', 'ao', 'bo', 'cn', 'co']

        if self.view not in valid_views:
            raise Exception('Current view is %s but channel %s can only be read from the views %s' %
                            (self.view, channel, valid_vis_views))

        rad = Dataset(join(self.path,channel+'_radiance_'+self.view+'.nc'))
        self.__image[channel] = rad.variables[channel+'_radiance_'+self.view][:]
        if self.__first:
            self.read_geometry(rad)

        self.__channels.add(channel)
        self.__first = False

    def _read_reflectance(self, channel):
        """Read in a given channel and convert the input to a reflectance"""

        self._read_solar_irradiance(channel)
        self.__image[channel+'r'] = self.radiance(channel) / (self.__solar_irradiance[channel] * self.__mu0) * np.pi
        self.__channels.add(channel+'r')


    def read_tir(self):

        valid_tir_views = ['in', 'io']

        if self.view in valid_tir_views:

            s7_bt = Dataset(self.path + 'S7_BT_' + self.view + '.nc')
            s7_bt.set_auto_scale(True)
            self.s7_image = s7_bt.variables['S7_BT_' + self.view][:]

            s8_bt = Dataset(self.path + 'S8_BT_' + self.view + '.nc')
            s8_bt.set_auto_scale(True)
            self.s8_image = s8_bt.variables['S8_BT_' + self.view][:]

            s9_bt = Dataset(self.path + 'S9_BT_' + self.view + '.nc')
            s9_bt.set_auto_scale(True)
            self.s9_image = s9_bt.variables['S9_BT_' + self.view][:] 
            
            self.__channels.update({'S7', 'S8', 'S9'})
            
            #print('read solar zenith (tir)')
            self.read_solar_zenith(s7_bt)

            self._fill_coords()
            self._read_flags()

    def read_fire(self):
        """ Read the fire channels"""
        valid_tir_views = ['in', 'io']
        if self.view in valid_tir_views:
            f1_bt = Dataset(self.path + 'F1_BT_' + self.view + '.nc')
            f1_bt.set_auto_scale(True)
            self.f1_image = f1_bt.variables['F1_BT_' + self.view][:] 
            f2_bt = Dataset(self.path + 'F2_BT_' + self.view + '.nc')
            f2_bt.set_auto_scale(True)
            self.f2_image = f2_bt.variables['F2_BT_' + self.view][:] 
            self.__channels.update({'F1', 'F2'})
            self._fill_coords()
            self._read_flags()

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

    def read_solar_zenith(self, cdata): # Seems just to be a subset of information in next routine.
        geometry = Dataset(join(self.path, 'geometry_tn.nc'))
        solar_zenith = geometry.variables['solar_zenith_tn'][:]
        solar_zenith_nonans = np.where(np.isfinite(solar_zenith), solar_zenith, 0.0)
        if cdata.start_offset == geometry.start_offset:
            start_offset = 0.0
        else:
            start_offset = cdata.start_offset * float(cdata.resolution.split()[2]) / float(geometry.resolution.split()[2]) - geometry.start_offset
        x = (np.array(range(cdata.dimensions['columns'].size)) - cdata.track_offset) * float(cdata.resolution.split()[1]) / float(geometry.resolution.split()[1]) + geometry.track_offset
        y = np.array(range(cdata.dimensions['rows'].size)) * float(cdata.resolution.split()[2]) / float(geometry.resolution.split()[2]) + start_offset
        t0 = time()
        f = interpolate.RectBivariateSpline(np.array(range(geometry.dimensions['rows'].size)), np.array(range(geometry.dimensions['columns'].size)), solar_zenith_nonans)
        self.solar_zenith = f(y, x)
        self.__mu0 = np.where(self.solar_zenith < 90, np.cos(self.__dtor * self.solar_zenith), 1.0)
        #print ('interpolation', time() - t0)

    def read_geometry(self, cdata):

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
        
    def plot(self, type='snow'):
        """Plot the data as different types. Allowed ones are 'snow' and 'vis'."""
        if type == 'snow':
            self._fill_rgb_snow()
        elif type == 'vis':
            self._fill_rgb_vis()
        else:
            print('Allowed types are "snow" and "vis"')
        p = plt.imshow(self.rgb)
        plt.show()
        
    def _fill_rgb_snow(self):
        """Fill rgb values for colour scheme highlighting snow"""
        ndsi = self.radiance('S5') - self.radiance('S1')
        r = np.ma.where(ndsi.data > 0, self.radiance('S5'), self.radiance('S3'))
        g = np.ma.where(ndsi.data > 0, self.radiance('S3'), self.radiance('S2'))
        b = np.ma.where(ndsi.data > 0, self.radiance('S2'), self.radiance('S1'))
        self.rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)

    def _fill_rgb_vis(self):
        """Fill rgb values for false colour image making ice clouds blue and liquid clouds white/pink"""
        for ch in {'S1', 'S2', 'S3'}: self._read_channel(ch)
        r = self.radiance('S1')
        g = self.radiance('S2')
        b = self.radiance('S3')
        self.rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)

    def _read_flags(self):
        """Reads the L1 cloud product variables""" 
        flags                = Dataset(join(self.path,'flags_' + self.view + '.nc'))
        self.cloud_flag      = flags.variables['cloud_'+self.view][:]  # Basic cloud flag. Contains a set of binary decisions
        self.confidence_flag = flags.variables['confidence_'+self.view][:] 
        self.pointing_flag   = flags.variables['pointing_'+self.view][:]
        self.bayesian_cloud  = flags.variables['bayes_'+self.view][:] # More fancy.

    def _fill_coords(self):
        """Reads the longitude and latitude information"""
 
        geodetic = Dataset(join(self.path,'geodetic_' + self.view + '.nc'))
        self.latitude  = geodetic['latitude_' + self.view][:]
        self.longitude = geodetic['longitude_' + self.view][:]

