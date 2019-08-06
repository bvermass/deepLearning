'''
This is a helper script for doing a parametrized model training for new physics searches
For explanation of the method see https://arxiv.org/abs/1601.07913
Read a parameter from the signal (such as a new physics particle mass) from a signal tree
Fill the background tree with randomly samples parameters from the total signal distribution 
'''

import numpy as np
import sys
import collections
from treeToArray import treeToArray
from ROOT import TFile, TTree, TBranch
from array import array



class ParameterAdder():

    def __init__( self, signal_tree, parameter_name ):
        self._parameter_name = parameter_name 

        parameter_array = treeToArray( signal_tree, parameter_name, cut = '' )
        parameter_counts = collections.Counter( parameter_array )
        total_count = len( parameter_array )
        
        self._parameters = []
        self._probabilities = []
        for parameter in parameter_counts:
            self._parameters.append( parameter )
            self._probabilities.append( float( parameter_counts[parameter] ) / total_count ) 
         

    #generate one or multiple random parameters 
    def _yieldRandomParameters( self, size = None ):
        return np.random.choice( self._parameters, p = self._probabilities, size = size )


    #add parameter tree
    def updateTree( self, tree_to_update ):
        num_events = tree_to_update.GetEntries()
        random_parameters = self._yieldRandomParameters( num_events )
        
        parameter_to_fill = np.array( 0., dtype = np.float32 )
        parameter_branch = tree_to_update.Branch( self._parameter_name, parameter_to_fill , '{}/F'.format( self._parameter_name) )
        for i, event in enumerate( tree_to_update ):
            tree_to_update.GetEntry( i )
            parameter_to_fill = np.array( random_parameters[i], dtype = np.float32 )
            parameter_branch.Fill()
        tree_to_update.Write()

#Class to use in case signal and background are contained in a single tree.
#signal and background will be recognized by the parameter value being 0(background) or a significant value(signal).
class ParameterAdderSingleTree():

    def __init__(self, tree, parameter_name ):
        self._parameter_name = parameter_name

        #get array that contains both signal and background
        parameter_array = treeToArray( tree, parameter_name, cut = '')

        #cut out background from array:
        parameter_array_signal = [parameter_value for parameter_value in parameter_array if parameter_value != 0]
        parameter_counts = collections.Counter( parameter_array_signal )
        total_count = len( parameter_array_signal )

        #store number of background events for updateTree
        self._num_events_background = len( parameter_array ) - len( parameter_array_signal )

        self._parameters = []
        self._probabilities = []
        for parameter in parameter_counts:
            self._parameters.append( parameter )
            self._probabilities.append( float( parameter_counts[parameter] ) / total_count )

    #generate one or multiple random parameters
    def _yieldRandomParameters( self, size = None ):
        return np.random.choice( self._parameters, p = self._probabilities, size = size )

    #rewrite parameter values for background (change 0 to random value)
    def cloneTreeAndParametrize( self, tree_to_read ):
        random_parameters = self._yieldRandomParameters( self._num_events_background )

        parameter_name_updated = self._parameter_name + '_parametrized'
        parameter_to_fill = array('i',[0])
        parameter_branch  = tree_to_read.Branch( parameter_name_updated, parameter_to_fill, '{}/I'.format( parameter_name_updated ) )

        i_background = 0
        for i in range(tree_to_read.GetEntries()):
            tree_to_read.GetEntry( i )

            if getattr( tree_to_read, self._parameter_name ) == 0:
                parameter_to_fill[0] = int(random_parameters[i_background])
                i_background += 1
            else:
                parameter_to_fill[0] = getattr(tree_to_read, self._parameter_name)
            parameter_branch.Fill()



	
if __name__ == '__main__' :
    
    if not( len( sys.argv ) >= 4 and len( sys.argv ) <= 6 ):
        print('Incorrect number of command line argument given. Aborting.')
        print('Usage : <python ParematerAddition.py root_file_name signal_tree_name background_tree_name parameter_name >')
        print('Or alternatively: <python ParematerAddition.py signal_root_file_name signal_tree_name background_root_file_name background_tree_name parameter_name >')
        print('Or for a single tree containing both signal and background: <python ParematerAddition.py signal_root_file_name combined_tree_name parameter_name >')
        sys.exit()

    else:
        
        # read trees from root file
        # signal and background are in the same tree
        if len( sys.argv ) == 4:
            root_file = TFile( sys.argv[1], 'update' )
            tree = root_file.Get( sys.argv[2] )

        # signal and background are in the same file but in different trees
        elif len( sys.argv ) == 5:
            root_file = TFile( sys.argv[1], 'update' )
            signal_tree = root_file.Get( sys.argv[2] )
            background_tree = root_file.Get( sys.argv[3] )
        
        #signal and background are in a different file
        else:
            signal_root_file = TFile( sys.argv[1] )
            signal_tree = signal_root_file.Get( sys.argv[2] )
            background_root_file = TFile( sys.argv[3], 'update' )
            background_tree = background_root_file.Get( sys.argv[4] )
        
        #make ParameterAdder and use it to add signal parameter to background tree
        parameter_name = sys.argv[len( sys.argv ) - 1]
        adder = ParameterAdder( signal_tree, parameter_name ) if len( sys.argv ) > 4 else ParameterAdderSingleTree( tree, parameter_name )
        adder.updateTree( background_tree ) if len( sys.argv ) > 4 else adder.cloneTreeAndParametrize( tree )
        root_file.Write("",TFile.kOverwrite)
        root_file.Close()
