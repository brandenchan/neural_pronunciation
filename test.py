import unittest
from data_handling import pad, DataIterator
from numpy.testing import assert_array_equal
import numpy as np

class Test(unittest.TestCase):
    # def test_batch(self):
    #     x = [[5, 7, 8, 9], [6, 3], [3], [1]]
    #     ret, lens = batch(x)
    #     assert_array_equal(ret, np.array([[5, 6, 3, 1],
    #                                       [7, 3, 0, 0],
    #                                       [8, 0, 0, 0],
    #                                       [9, 0, 0, 0]]))
    #     assert lens == [4,2,1,1]

    def test_data_iterator(self):
        di = DataIterator(range(10))
        result = di.next(5)        
        assert result == list(range(5))

    def test_data_iterator_reset(self):
        di = DataIterator(range(5))
        result = di.next(10)
        assert result == [0,1,2,3,4,0,1,2,3,4]

    def test_data_iterator_stop(self):
        di = DataIterator(range(5), reset=False)
        result = di.next(10)
        assert result == [0,1,2,3,4]

    def test_pad(self):
        x = [[5, 7, 8, 9], [6, 3], [3], [1]]
        result = pad(x)
        assert result == [[5, 7, 8, 9], 
                          [6, 3, '<PAD>', '<PAD>'], 
                          [3, '<PAD>', '<PAD>', '<PAD>'], 
                          [1, '<PAD>', '<PAD>', '<PAD>']]


if __name__ == '__main__':
    unittest.main()
