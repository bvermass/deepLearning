import numpy as np

#import other parts of framework
import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert(0, main_directory )
from randomizer.shuffle import shuffleSimulataneously
from randomizer.eventSize import numberOfEventsToRead


#determine the shape of a branch 
def branchShape( uproot_branch ):
    return uproot_branch.lazyarray().shape


class PFNSampleGenerator :

    def __init__( self, uproot_tree, pfn_branch_names, highlevel_branch_names, label_branch_name, validation_fraction = 0.0, test_fraction = 0.0):
        self.__tree = uproot_tree
        self.__pfn_branches = [ uproot_tree[branch] for branch in pfn_branch_names ]
        self.__highlevel_branches = [ uproot_tree[branch] for branch in highlevel_branch_names ]
        self.__label_branch = uproot_tree[label_branch_name]
        self.__validation_fraction = validation_fraction
        self.__test_fraction = test_fraction


    def __len__(self):
        return len( self.__tree )


    #determine maximum number of events to read at once (using maximum 1 GB of ram or 50K events)
    def __maxArraySize( self, maximum_amount = 50000, memory_limit = 1 ):
        return numberOfEventsToRead( self.__tree, maximum_amount, memory_limit )


    #make array for particle-flow inputs 
    def __pfnArray( self, startIndex, stopIndex ):
        number_of_objects = branchShape( self.__pfn_branches[0] )[-1]
        
        #add extra dimension to each array and shift it to the middle
        #the different features will be added along this dimension
        pfn_arrays = list( np.expand_dims( pfn_branch.array( entrystart = startIndex, entrystop = stopIndex ), -1 ).reshape( stopIndex - startIndex, 1, number_of_objects ) for pfn_branch in self.__pfn_branches )
        
        #merge the objects along the middle axis so the array has shape  (number_of_events, number_of_features, number_of_objects )
        pfn_samples = np.concatenate( pfn_arrays, axis = 1 )
        
        #swap the feature and object axes so the array contains a list of objects with all of their respective features for each event
        pfn_samples = np.transpose( pfn_samples, axes = (0, 2, 1) )
        
        return pfn_samples


    def __highlevelArray( self, startIndex, stopIndex ):

        #add extra dimension along which to add features
        highlevel_arrays = list( np.expand_dims( highlevel_branch.array( entrystart = startIndex, entrystop = stopIndex ), -1 ) for highlevel_branch in self.__highlevel_branches )
        highlevel_samples = np.concatenate( highlevel_arrays, axis = -1 )
        
        return highlevel_samples


    def __labelArray( self, startIndex, stopIndex ):
        return self.__label_branch.array( entrystart = startIndex, entrystop = stopIndex )


    #generate training samples
    def __sampleGenerator( self, min_index, max_index, batch_size):

        max_array_size = self.__maxArraySize()

        #adjust the array size to be a divisor of the batch size
        max_array_size -= max_array_size%batch_size 

        #keep yielding events indefinitely to allow an arbitrary number of training epochs
        while True:

            for split_index in range( min_index, max_index, max_array_size ):
                array_length = min( max_array_size, max_index - split_index )

                #feature samples and labels
                pfn_samples = self.__pfnArray( split_index, split_index + array_length )
                highlevel_samples = self.__highlevelArray( split_index, split_index + array_length )
                labels = self.__labelArray( split_index, split_index + array_length )

                #shuffle arrays 
                shuffleSimulataneously( pfn_samples, highlevel_samples, labels )

                #yield batches 
                for batch_index in range( 0, array_length, batch_size ):
                    yield [ pfn_samples[ batch_index : batch_index + batch_size ], highlevel_samples[ batch_index : batch_index + batch_size ] ], labels[ batch_index : batch_index + batch_size ]


    def __minTrainingIndex( self ):
        return 0


    def __minValidationIndex( self ):
        return int( len( self.__tree )*( 1 - self.__validation_fraction - self.__test_fraction ) )


    def __minTestIndex( self ):
        return int( len( self.__tree )*( 1 - self.__test_fraction ) )


    #training set generator 
    def trainingGenerator( self, batch_size ):
        return self.__sampleGenerator( self.__minTrainingIndex(), self.__minValidationIndex(), batch_size )


    #validation set generator
    def validationGenerator( self, batch_size ):
        return self.__sampleGenerator( self.__minValidationIndex(), self.__minTestIndex(), batch_size )


    #test set generator
    def testGenerator( self, batch_size ):
        return self.__sampleGenerator( self.__minTestIndex(), len( self.__tree ), batch_size )

    
    def __numberOfBatches( self, total_size, batch_size ):
        return int( total_size / batch_size ) + ( total_size % batch_size )


    #number of training batches 
    def numberOfTrainingBatches( self, batch_size ):
        return self.__numberOfBatches( self.__minValidationIndex() - self.__minTrainingIndex(), batch_size )


    #number of validation batches 
    def numberOfValidationBatches( self, batch_size ):
        return self.__numberOfBatches( self.__minTestIndex() - self.__minValidationIndex(), batch_size )

    
    #number of testing batches 
    def numberOfTestBatches( self, batch_size ):
        return self.__numberOfBatches( len( self.__tree ) - self.__minTestIndex(), batch_size )



if __name__ == '__main__':
    pass
