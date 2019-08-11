#name of input root file, relative to the directory of this script
root_file_name = '/user/bvermass/public/2l2q_analysis/trees/HNLBDTtagger/final/full_analyzer/HNLBDTtagger_muon_randomized_xgboost.root'

signal_tree_name = 'signalTree'
background_tree_name = 'bkgTree'

#list of variables to be used in training (corresponding to branches in the tree)
list_of_branches = [
    '_gen_Nmass_parametrized',
    '_lPt', '_lEta', '_lPhi', '_ldxy', '_ldz', '_l3dIPSig', '_lrelIso', '_lptRel', '_lptRatio', '_lNumberOfPixelHits',
    '_JetPt', '_JetEta', '_JetPhi', '_JetMass', '_nJetConstituents', '_JetdxySum', '_JetdxySigSum', '_JetdzSum', '_JetdzSigSum', '_JetChargeSum',
    '_SV_ntracks', '_SV_PVSVdist_2D', '_SV_PVSVdist', '_SV_normchi2', '_SV_mass', '_SV_pt', '_SV_eta', '_SV_phi'
    ]

weight_branch = '_weight'

only_positive_weights = True

validation_fraction = 0.4
test_fraction = 0.2

number_of_threads = 1

use_genetic_algorithm = True
high_memory = False

if use_genetic_algorithm:

    population_size = 100

    #ranges of neural network parameters for the genetic algorithm to scan
    parameter_ranges = {
        'number_of_trees' : list( range(100, 3000) ),
		'learning_rate' : (0.001, 1),
		'max_depth' : list( range(2, 11) ),
		'min_child_weight' : (1, 20),
		'subsample' : (0.1, 1),
		'colsample_bytree' : (0.5, 1),
		'gamma' : (0, 1),
		'alpha' : (0, 1)
	}

else:
    parameter_values = {
        'number_of_trees' : [500, 1000, 2000, 4000, 8000],
		'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.5],
		'max_depth' : [2, 3, 4, 5, 6],
		'min_child_weight' : [1, 5, 10],
		'subsample' : [1],
		'colsample_bytree' : [0.5, 1],
		'gamma' : [0],
		'alpha' : [0]
	}

