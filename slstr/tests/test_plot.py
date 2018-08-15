try:
    import unittest.mock as mock
except ImportError:
    import mock
from unittest import TestCase

import slstr.reader

import slstr.plotter

def mock_reader():
    """Create a mock reader that the tests can use"""
    import numpy as np
    from collections import defaultdict
    a = np.array([
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 3.0, 3.0, 4.0]])
    return mock.MagicMock(radiance=lambda s: a)

class TestPlotter(TestCase):
        
    @mock.patch('slstr.reader.Reader')
    @mock.patch('slstr.plotter.plt')
    def test_vis(self, mocked_plt, mocked_Reader):
        r = slstr.reader.Reader('xyz')
        slstr.plotter.vis(r)
        mocked_plt.imshow.assert_called_once()
        mocked_plt.show.assert_called_once()

    @mock.patch('slstr.reader.Reader')
    @mock.patch('slstr.plotter.plt')
    def test_snow(self, mocked_plt, mocked_Reader):
        mocked_Reader.return_value = mock_reader()
        r = slstr.reader.Reader('xyz')
        slstr.plotter.snow(r)
        mocked_plt.imshow.assert_called_once()
        mocked_plt.show.assert_called_once()

    @mock.patch('slstr.reader.Reader')
    @mock.patch('slstr.plotter.plt')
    def test_radiance(self, mocked_plt, mocked_Reader):
        r = slstr.reader.Reader('xyz')
        slstr.plotter.radiance(r, 'A', 'B', 'C')
        mocked_plt.imshow.assert_called_once()
        mocked_plt.show.assert_called_once()

    @mock.patch('slstr.reader.Reader')
    @mock.patch('slstr.plotter.plt')
    def test_reflectance(self, mocked_plt, mocked_Reader):
        r = slstr.reader.Reader('xyz')
        slstr.plotter.reflectance(r, 'A', 'B', 'C')
        mocked_plt.imshow.assert_called_once()
        mocked_plt.show.assert_called_once()
