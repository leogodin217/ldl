from ldl.utitlities import *
import numpy as np


def test_normalize_2d_array_works_with_valid_date():
    data = np.array([
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [1, 8],
        [1, 9],
        [1, 10],
    ])

    normalized = normalize_2d_array(data)
    # Check shape
    normalized.should.be.a(np.ndarray)
    normalized.shape.should.equal((11, 2))
    # column with all samve values should be all zeros
    normalized[0, 0].should.equal(0)
    normalized[1, 0].should.equal(0)
    normalized[10, 0].should.equal(0)
    # Simply case where normalized in number / 10
    # There is a little noise, because we add a small amount to the denominator
    # in case there is a zero denominator
    normalized[0, 1].should.equal(0)
    normalized[1, 1].should.be.within(0.099, 0.1)
    normalized[2, 1].should.be.within(0.199, 0.2)
    normalized[10, 1].should.be.within(0.99, 0.1)
