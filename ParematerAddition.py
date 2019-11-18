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

    def __init__( self, signal_tree, parameter_names ):
        self._parameter_names = parameter_names

        parameter_arrays = treeToArray( signal_tree, parameter_names, cut = '' )
        parameter_counts = collections.Counter( tuple(x) for x in parameter_arrays )
        total_count = len( parameter_arrays )
        
        self._parameters = []
        self._probabilities = []
        for parameters in parameter_counts:
            self._parameters.append( parameters )
            self._probabilities.append( float( parameter_counts[parameters] ) / total_count )


    #generate one or multiple random indices, based on an ordered list self._parameters (giving the ordered list itself as argument doesn't work since the elements of the list are tuples)
    def _yieldRandomParameters( self, size = None ):
        return np.random.choice( np.arange(len(self._parameters)), p = self._probabilities, size = size )


    #add parameter tree
    def updateTree( self, tree_to_update ):
        num_events = tree_to_update.GetEntries()
        random_parameter_index = self._yieldRandomParameters( num_events )

        parameters_to_fill = {}
        parameter_branches = {}
        for parameter_name in self._parameter_names:
            parameters_to_fill[parameter_name] = array( 'd', [0] )
            parameter_branches[parameter_name] = tree_to_update.Branch( parameter_name + "_parametrized", parameters_to_fill[parameter_name] , '{}/D'.format( parameter_name + "_parametrized") )

        for i, event in enumerate( tree_to_update ):
            tree_to_update.GetEntry( i )

            for parameter_name in self._parameter_names:
                parameters_to_fill[parameter_name][0] = self._parameters[random_parameter_index[i]][self._parameter_names.index(parameter_name)]
                parameter_branches[parameter_name].Fill()
        tree_to_update.Write()

#Class to use in case signal and background are contained in a single tree.
#signal and background will be recognized by the parameter value being 0(background) or a significant value(signal).
class ParameterAdderSingleTree():

    def __init__(self, tree, parameter_names ):
        self._parameter_names = parameter_names

        #get arrays containing the variables
        parameter_arrays = treeToArray( tree, parameter_names, cut = '')

        #cut out background from arrays:
        parameter_arrays_signal = [parameter_value for parameter_value in parameter_arrays if not parameter_value.all() == 0]
        parameter_counts = collections.Counter(tuple(x) for x in parameter_arrays_signal)
        total_count = len( parameter_arrays_signal )

        #store number of background events for updateTree
        self._num_events_background = len( parameter_arrays ) - len( parameter_arrays_signal )

        self._parameters = []
        self._probabilities = []
        for parameters in parameter_counts:
            self._parameters.append( parameters )
            self._probabilities.append( float( parameter_counts[parameters] ) / total_count )

    #generate one or multiple random parameter indices, based on an ordered list (giving the ordered list itself doesn't work since the elements of the list are tuples)
    def _yieldRandomParameterIndex( self, size = None ):
        return np.random.choice( np.arange(len(self._parameters)), p = self._probabilities, size = size )

    #rewrite parameter values for background (change 0 to random value)
    def cloneTreeAndParametrize( self, tree ):
        random_parameter_index = self._yieldRandomParameterIndex( self._num_events_background )

        parameters_to_fill = {}
        parameter_branches = {}
        i_background       = {}
        for parameter_name in self._parameter_names:
            parameters_to_fill[parameter_name] = array( 'd', [0] )
            parameter_branches[parameter_name] = tree.Branch( parameter_name + '_parametrized', parameters_to_fill[parameter_name], '{}/D'.format( parameter_name + '_parametrized' ) )
            i_background[parameter_name] = 0

        for i in range(tree.GetEntries()):
            tree.GetEntry( i )

            for parameter_name in self._parameter_names:
                # value == 0 means background event, change value to random one
                if getattr( tree, parameter_name ) == 0:
                    parameters_to_fill[parameter_name][0] = self._parameters[random_parameter_index[i_background[parameter_name]]][self._parameter_names.index(parameter_name)]
                    i_background[parameter_name] += 1
                # value != 0 means signal event, don't change value from original one
                else:
                    parameters_to_fill[parameter_name][0] = getattr(tree, parameter_name)
                parameter_branches[parameter_name].Fill()
        tree.Write()



if __name__ == '__main__' :
    
    if not( len( sys.argv ) >= 4 and len( sys.argv ) <= 6 ):
        print('Incorrect number of command line argument given. Aborting.')
        print('Usage : <python ParematerAddition.py root_file_name signal_tree_name background_tree_name parameter_name,parameter_name2,... >')
        print('Or alternatively: <python ParematerAddition.py signal_root_file_name signal_tree_name background_root_file_name background_tree_name parameter_name,parameter_name2,... >')
        print('Or for a single tree containing both signal and background: <python ParematerAddition.py signal_root_file_name combined_tree_name parameter_name,parameter_name2,... >')
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
        
        #make ParameterAdder and use it to add signal parameters to background tree (parameter_names is an array of the parameters that need to be added in a correlated way)
        parameter_names = sys.argv[len( sys.argv ) - 1].split(',')
        adder = ParameterAdder( signal_tree, parameter_names ) if len( sys.argv ) > 4 else ParameterAdderSingleTree( tree, parameter_names )
        adder.updateTree( background_tree ) if len( sys.argv ) > 4 else adder.cloneTreeAndParametrize( tree )

        if len( sys.argv ) < 6:
            root_file.Write("",TFile.kOverwrite)
            root_file.Close()
        else:
            signal_root_file.Write("",TFile.kOverwrite)
            signal_root_file.Close()
            background_root_file.Write("",TFile.kOverwrite)
            background_root_file.Close()
