import uproot

#import other parts of framework
import os
import sys
import argparse
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert(0, main_directory )
from DeepSets.PFN import PFN
from DeepSets.PFNSampleGenerator import PFNSampleGenerator
from modelTraining.trainKerasModel import tensorFlowSetNumThreads
from runTraining import stringToArgumentType


if __name__ == '__main__':
    
    #This limits the number of cores that keras uses to one (meant for submitting jobs otherwise keras takes as many cores as it can which makes it crash)
    tensorFlowSetNumThreads(1)
    
    #read root file and tree using uproot
    #fill in your file and tree names here
    #root_file_name = '~/Work/jetTagger/SampleGenerator/mergedFile_randomized.root'
    #root_file_name = '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/mergedFile_randomized.root'
    if len( sys.argv ) == 2 :
        root_file_name = sys.argv[1]
        hidden_layers_latent = 2
        nodes_latent = 128
        dropout_latent = 0.1
        latent_space = 512
        hidden_layers_output = 2
        nodes_output = 128
        dropout_output = 0.1
        learning_rate = 1
        learning_rate_decay = 1
    elif len( sys.argv ) == 11 :
        root_file_name = sys.argv[1]
        configuration_dict = {}
        for argument in sys.argv[2:] :
            if not '=' in argument:
                raise ValueError('Each command line must be of the form x=y in order to correctly parse them')
            key, value = argument.split('=')
            configuration_dict[key] = stringToArgumentType( value )

    else :
        print('Error: invalid number of arguments')
    tree_name = 'HNLtagger_tree'

    f = uproot.open( root_file_name )
    tree = f[ tree_name ]
    
    #make a Sample generator 
    #fill in the names of your particle-flow and high level branch names here
    pfn_branch_names = [ '_JetConstituentPt', '_JetConstituentEta', '_JetConstituentPhi', '_JetConstituentdxy', '_JetConstituentdz', '_JetConstituentdxyErr', '_JetConstituentdzErr', '_JetConstituentNumberOfHits', '_JetConstituentNumberOfPixelHits', '_JetConstituentCharge', '_JetConstituentPdgId', '_JetConstituentInSV']
    highlevel_branch_names = [ '_JetPt', '_JetEta', '_SV_PVSVdist', '_SV_PVSVdist_2D', '_SV_ntracks', '_SV_mass', '_SV_pt', '_SV_eta', '_SV_phi', '_SV_normchi2' ]
    label_branch = '_JetIsFromHNL'
    
    sample = PFNSampleGenerator( tree, pfn_branch_names, highlevel_branch_names, label_branch, validation_fraction = 0.4, test_fraction = 0.2 )
    
    #set up the neural network with default arguments
    network = PFN( (50, 12), ( 10, ), 
            num_hidden_layers_latent = configuration_dict['hidden_layers_latent'],
            nodes_per_layer_latent = configuration_dict['nodes_latent'],
            batch_normalization_latent = True,
            dropout_rate_latent = configuration_dict['dropout_latent'],
            latent_space_size = configuration_dict['latent_space'],
            num_hidden_layers_output = configuration_dict['hidden_layers_output'],
            nodes_per_layer_output = configuration_dict['nodes_latent'],
            batch_normalization_output = True,
            dropout_rate_output = configuration_dict['dropout_output'],
            optimizer_name = 'Nadam',
            relative_learning_rate = configuration_dict['learning_rate'],
            relative_learning_rate_decay = configuration_dict['learning_rate_decay']
    )

    #train with default arguments
    network.trainModel( sample, 'jetTagger.h5') 
