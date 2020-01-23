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
    tensorFlowSetNumThreads(8)
    
    #read root file and tree using uproot
    #fill in your file and tree names here
    #root_file_name = '~/Work/jetTagger/SampleGenerator/mergedFile_randomized.root'
    #root_file_name = '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/mergedFile_randomized.root'
    if len( sys.argv ) == 2 :
        root_file_name = sys.argv[1]
        configuration_dict = {}
        configuration_dict['hidden_layers_latent'] = 3
        configuration_dict['nodes_latent'] = 128
        configuration_dict['dropout_latent'] = 0.5
        configuration_dict['latent_space'] = 256
        configuration_dict['hidden_layers_output'] = 3
        configuration_dict['nodes_output'] = 128
        configuration_dict['dropout_output'] = 0.5
        configuration_dict['learning_rate'] = 1
        configuration_dict['learning_rate_decay'] = 1
        configuration_dict['activation_name'] = 'prelu'
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
    highlevel_branch_names = [ '_JetPt_log', '_JetEta', '_JetPhi', '_lPt', '_lEta', '_lPhi', '_ldxy_sgnlog', '_ldz_sgnlog', '_l3dIPSig', '_lrelIso', '_lptRel', '_lptRatio', '_lNumberOfPixelHits', '_SV_PVSVdist', '_SV_PVSVdist_2D', '_SV_ntracks', '_SV_mass', '_SV_pt', '_SV_eta', '_SV_phi', '_SV_normchi2', '_SV_l1mass', '_nJetConstituents', '_l1Pt_log', '_l1Eta', '_mll', '_dRll', '_dRljet' ]
    pfn_branch_names = [ '_JetConstituentPt_log', '_JetConstituentEta', '_JetConstituentPhi', '_JetConstituentPdgId', '_JetConstituentCharge', '_JetConstituentdxy_sgnlog', '_JetConstituentdxyErr', '_JetConstituentdz_sgnlog', '_JetConstituentdzErr', '_JetConstituentNumberOfHits', '_JetConstituentNumberOfPixelHits', '_JetConstituentHasTrack', '_JetConstituentInSV' ]


    label_branch = '_JetIsFromHNL'

    sample = PFNSampleGenerator( tree, pfn_branch_names, highlevel_branch_names, label_branch, validation_fraction = 0.3, test_fraction = 0.1, parameter_branch_names = ['_gen_Nmass', '_gen_Nctau'], parameter_background_defaults = [0,0] )

    #set up the neural network with default arguments
    network = PFN( (50, 13), ( 28, ), num_hidden_layers_latent = 3, nodes_per_layer_latent = 128, batch_normalization_latent = True, dropout_rate_latent = 0.5, latent_space_size = 256, num_hidden_layers_output = 3, nodes_per_layer_output = 128, batch_normalization_output = True, dropout_rate_output = 0.5, optimizer_name = 'Nadam', relative_learning_rate = 1, relative_learning_rate_decay = 1, activation_name = 'prelu' )

    
    #set up the neural network with default arguments
    #maybe resend Willem these changes I made on top of his
    #network = PFN( (50, 13), ( 23, ),
    #        num_hidden_layers_latent = configuration_dict['hidden_layers_latent'],
    #        nodes_per_layer_latent = configuration_dict['nodes_latent'],
    #        batch_normalization_latent = True,
    #        dropout_rate_latent = configuration_dict['dropout_latent'],
    #        latent_space_size = configuration_dict['latent_space'],
    #        num_hidden_layers_output = configuration_dict['hidden_layers_output'],
    #        nodes_per_layer_output = configuration_dict['nodes_latent'],
    #        batch_normalization_output = True,
    #        dropout_rate_output = configuration_dict['dropout_output'],
    #        optimizer_name = 'Nadam',
    #        relative_learning_rate = configuration_dict['learning_rate'],
    #        relative_learning_rate_decay = configuration_dict['learning_rate_decay']
    #        activation_name = configuration_dict['activation_name']
    #)

    #train with default arguments
    network.trainModel( sample, 'jetTagger.h5', 512, 2000) 
