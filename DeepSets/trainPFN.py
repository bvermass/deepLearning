import uproot

#import other parts of framework
import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert(0, main_directory )
from DeepSets.PFN import PFN
from DeepSets.PFNSampleGenerator import PFNSampleGenerator
from modelTraining.trainKerasModel import tensorFlowSetNumThreads



if __name__ == '__main__':
    
    #This limits the number of cores that keras uses to one (meant for submitting jobs otherwise keras takes as many cores as it can which makes it crash)
    tensorFlowSetNumThreads(1)
    
    #read root file and tree using uproot
    #fill in your file and tree names here
    #root_file_name = '~/Work/jetTagger/SampleGenerator/mergedFile_randomized.root'
    #root_file_name = '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/mergedFile_randomized.root'
    if len( sys.argv ) == 2 :
        root_file_name = sys.argv[1]
    else :
        print('Error: invalid number of arguments')
    tree_name = 'HNLtagger_tree'

    f = uproot.open( root_file_name )
    tree = f[ tree_name ]
    
    #make a Sample generator 
    #fill in the names of your particle-flow and high level branch names here
    pfn_branch_names = [ '_JetConstituentPt', '_JetConstituentEta', '_JetConstituentPhi', '_JetConstituentdxy', '_JetConstituentdz', '_JetConstituentdxyErr', '_JetConstituentdzErr', '_JetConstituentNumberOfHits', '_JetConstituentNumberOfPixelHits', '_JetConstituentCharge', '_JetConstituentPdgId']
    highlevel_branch_names = [ '_JetPt', '_JetEta' ]
    label_branch = '_JetIsFromHNL'
    
    sample = PFNSampleGenerator( tree, pfn_branch_names, highlevel_branch_names, label_branch, validation_fraction = 0.4, test_fraction = 0.2 )
    
    #set up the neural network with default arguments
    network = PFN( (50, 11), ( 2, ) )

    #train with default arguments
    network.trainModel( sample, 'jetTagger_reliso_novtx.h5') 
