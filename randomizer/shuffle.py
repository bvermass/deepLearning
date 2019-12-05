"""
Tools for shuffling arrays
"""

import numpy as np

def randomIndices( length ):
    return np.random.permutation( length )


#shuffle several arrays consistencly
def shuffleSimulataneously( *arrays ):

    #check that all input arrays have the same length
    array_size = len( arrays[0] )
    if not( all( len(array) == array_size for array in arrays ) ):
        raise ValueError('All arrays must have the same length to be simultaneously shuffled.')

    #shuffle arrays
    random_indices = randomIndices( array_size )
    for array in arrays:
        array = array[ random_indices ]
