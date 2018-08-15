try:
    import unittest.mock as mock
except ImportError:
    import mock
from unittest import TestCase

from slstr.reader import Reader

def mock_dataset():
    """Create a mock netCDF4 dataset that the tests can use"""
    import numpy as np
    from collections import defaultdict
    a = np.array([
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 3.0, 3.0, 4.0]])
    return mock.MagicMock(variables=defaultdict(lambda: a),
                          __getitem__= lambda i,j: a,
                          dimensions =
                            defaultdict(lambda: np.array([0,1,2,3,4])),
                          start_offset=0,
                          track_offset=0,
                          resolution='1 \n1 \n1')

class TestReader(TestCase):

    def test_constructor(self):
        r = Reader('xyz')

    @mock.patch('slstr.reader.Reader.read_channel')
    def test_read_argument(self,mocked_read):
        r = Reader('xyz')
        r.read_channel('fake')
        mocked_read.assert_called_once_with('fake')
    
    def test_invalid_channel(self):
        r = Reader('xyz')
        self.assertRaises(Exception, r.read_channel, 'invalid')

    @mock.patch('slstr.reader.Dataset')
    def test_Dataset(self, mocked_Dataset):
        mocked_Dataset.return_value = mock_dataset()
        r = Reader('xyz')
        r._fill_coords()
        
    @mock.patch('slstr.reader.Dataset')
    def test_read_channel(self, mocked_Dataset):
        mocked_Dataset.return_value = mock_dataset()
        r = Reader('xyz')
        r.read_channel('S1')

    @mock.patch('slstr.reader.Dataset')
    def test_read_reflectance(self, mocked_Dataset):
        mocked_Dataset.return_value = mock_dataset()
        r = Reader('xyz')
        r.read_reflectance('S1')

        
    @mock.patch('slstr.reader.Dataset')
    def test_caching(self, mocked_Dataset):
        mocked_Dataset.return_value = mock_dataset()
        r = Reader('xyz')
        r.read_channel('S1')

        # Test that caching works
        mocked_Dataset.reset_mock()
        r.read_channel('S1')
        mocked_Dataset.assert_not_called()

        # Test no caching if different channel
        mocked_Dataset.reset_mock()
        r.read_channel('S2')
        mocked_Dataset.assert_called()

    @mock.patch('slstr.reader.Dataset')
    @mock.patch('slstr.reader.plt')
    def test_plot(self, mocked_plt, mocked_Dataset):
        mocked_Dataset.return_value = mock_dataset()
        r = Reader('xyz')
        r.plot('vis')
        mocked_plt.imshow.assert_called_once()
        mocked_plt.show.assert_called_once()

        mocked_plt.reset_mock()
        r.plot()
        mocked_plt.imshow.assert_called_once()
        mocked_plt.show.assert_called_once()
        
        mocked_plt.reset_mock()
        r.plot('invalid')
        mocked_plt.assert_not_called()
