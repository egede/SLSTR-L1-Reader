# 
# Copyright: Imperial College London.
# Licence  : GPLv3
# Created  : August 2018
#
import numpy as np
import matplotlib.pyplot as plt

def radiance(reader, rchannel, gchannel , bchannel):
    """Fill rgb values for a false colour image with the channels given as strings"""
    r = reader.radiance(rchannel)
    g = reader.radiance(gchannel)
    b = reader.radiance(bchannel)
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    p = plt.imshow(rgb)
    plt.show()

def reflectance(reader, rchannel, gchannel , bchannel):
    """Fill rgb values for a false colour image with the channels given as strings"""
    r = reader.reflectance(rchannel)
    g = reader.reflectance(gchannel)
    b = reader.reflectance(bchannel)
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    p = plt.imshow(rgb)
    plt.show()
    
def snow(reader):
    """Fill rgb values for colour scheme highlighting snow"""
    ndsi = reader.radiance('S5') - reader.radiance('S1')
    r = np.ma.where(ndsi.data > 0, reader.radiance('S5'), reader.radiance('S3'))
    g = np.ma.where(ndsi.data > 0, reader.radiance('S3'), reader.radiance('S2'))
    b = np.ma.where(ndsi.data > 0, reader.radiance('S2'), reader.radiance('S1'))
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    p = plt.imshow(rgb)
    plt.show()
    
def vis(reader):
    """Fill rgb values for false colour image making ice clouds blue and liquid clouds white/pink"""
    radiance(reader, 'S1', 'S2', 'S3')

