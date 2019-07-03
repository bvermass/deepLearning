import numpy as np
import uproot
from keras import models, layers, Input, Model
from keras import backend as K
import matplotlib.pyplot as plt


from PFNSampleGenerator import PFNSampleGenerator
from diagnosticPlotting import *



def getOutputArray( sample_generator, size, model):
    outputs = None
    labels = None
    i = 0
    for sample_batch, labels_batch in sample_generator:
        prediction_batch = model.predict( sample_batch )
        prediction_batch = np.reshape( prediction_batch, len( prediction_batch ) )
        
        if outputs is None:
            outputs = prediction_batch
            labels = labels_batch
        else:
            outputs = np.concatenate( [outputs, prediction_batch], axis = -1 )
            labels = np.concatenate( [labels, labels_batch], axis = -1 )

        i += 1
        if i >= size:
            break
    return outputs, labels


def signalBackgroundSplit( outputs, labels ):
    signal = outputs[ labels == True ]
    background = outputs[ labels == False ]
    return signal, background


def plotShapeComparison( signal_training_outputs, background_training_outputs, signal_validation_outputs, background_validation_outputs, name, log):
    min_output = min( np.min(signal_training_outputs), np.min( background_training_outputs ), np.min( signal_validation_outputs ), np.min( background_validation_outputs ) )
    max_output = max( np.max(signal_training_outputs), np.max( background_training_outputs ), np.max( signal_validation_outputs ), np.max( background_validation_outputs ) )
    
    addHist( signal_training_outputs, np.ones( len(signal_training_outputs) ) , 30, min_output, max_output, 'Signal training set', color='blue')
    addHist( background_training_outputs, np.ones( len( background_training_outputs ) ), 30, min_output, max_output, 'Background training set', color = 'red')
    addHist( signal_validation_outputs, np.ones( len( signal_validation_outputs ) ), 30, min_output, max_output, 'Signal validation set', color='purple')
    addHist( background_validation_outputs, np.ones( len( background_validation_outputs ) ), 30, min_output, max_output, 'Background validation set', color='green')
    
    plt.xlabel( 'Model output', fontsize = 22 )
    plt.ylabel( 'Normalized events', fontsize = 22 )
    if log:
    	plt.yscale( 'log' )
    plt.legend(ncol=2, prop={'size': 10})
    
    bottom, top = plt.ylim()
    if log:
        plt.ylim( bottom, top*10 )
    else:
        plt.ylim( 0,  top*1.2)
    plt.savefig(name if '.' in name else name + '.pdf')
    plt.clf() 


def generateModel():
    def summation(x):
        x = K.sum( x, axis = 1 )
        return x 

    jet_size = 50
    pfn_input = Input( shape = (jet_size, 11 ) )
    pfn_intermediate = layers.Masking( mask_value = 0 )( pfn_input )
    pfn_intermediate = layers.BatchNormalization()( pfn_intermediate )
    pfn_intermediate = layers.TimeDistributed( layers.Dense( 128, activation = 'relu' ) )( pfn_intermediate )
    pfn_intermediate = layers.BatchNormalization()( pfn_intermediate )
    pfn_intermediate = layers.TimeDistributed( layers.Dense( 128, activation = 'relu' ) )( pfn_intermediate )
    pfn_intermediate = layers.BatchNormalization()( pfn_intermediate )
    pfn_intermediate = layers.TimeDistributed( layers.Dense( 256, activation = 'linear' ) )( pfn_intermediate )
    pfn_intermediate = layers.Lambda( summation, output_shape=None, mask=None, arguments=None)( pfn_intermediate )
    
    highlevel_input = Input( shape = (2, ) )
    
    merged_intermediate = layers.concatenate( [pfn_intermediate, highlevel_input], axis = -1 )
    merged_intermediate = layers.BatchNormalization()( merged_intermediate )
    merged_intermediate = layers.Dense( 128, activation = 'relu' )( merged_intermediate )
    merged_intermediate = layers.BatchNormalization()( merged_intermediate )
    merged_intermediate = layers.Dense( 128, activation = 'relu' )( merged_intermediate )
    merged_output = layers.Dense( 1, activation = 'sigmoid' )( merged_intermediate )
    model = Model( [pfn_input, highlevel_input] , merged_output )
    return model 



def makeOutputArrays():
    f = uproot.open('/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/mergedFile_randomized.root')
    tree = f['HNLtagger_tree']
    pfn_branch_names = [ '_JetConstituentPt', '_JetConstituentEta', '_JetConstituentPhi', '_JetConstituentdxy', '_JetConstituentdz', '_JetConstituentdxyErr', '_JetConstituentdzErr', '_JetConstituentNumberOfHits', '_JetConstituentNumberOfPixelHits', '_JetConstituentCharge', '_JetConstituentPdgId']
    highlevel_branch_names = [ '_JetPt', '_JetEta' ]
    label_branch = '_JetIsFromHNL'
    
    
    sample_generator = PFNSampleGenerator( tree, pfn_branch_names, highlevel_branch_names, label_branch, validation_fraction = 0.4, test_fraction = 0.2 )
    model = generateModel()
    model.load_weights( '/user/bvermass/public/PFN/JetTagger/jetTagger_reliso_novtx.h5' )
    
    training_generator = sample_generator.trainingGenerator( 512 )
    training_outputs, training_labels = getOutputArray( training_generator, sample_generator.numberOfTrainingBatches( 512 ), model )
    signal_training_outputs, background_training_outputs = signalBackgroundSplit( training_outputs, training_labels)
    
    np.save( 'signal_training_outputs.npy', signal_training_outputs )
    np.save( 'background_training_outputs.npy', background_training_outputs )
    
    print( 'len(signal_training_outputs) = {}'.format( len(signal_training_outputs) ) )
    
    validation_generator = sample_generator.validationGenerator( 512 )
    validation_outputs, validation_labels = getOutputArray( validation_generator, sample_generator.numberOfValidationBatches( 512 ), model )
    signal_validation_outputs, background_validation_outputs = signalBackgroundSplit( validation_outputs, validation_labels)
    
    np.save( 'signal_validation_outputs.npy', signal_validation_outputs )
    np.save( 'background_validation_outputs.npy', background_validation_outputs )

    print( 'len(signal_validation_outputs) = {}'.format( len(signal_validation_outputs) ) )


def producePlots():
    signal_training_outputs = np.load( 'signal_training_outputs.npy' )
    background_training_outputs = np.load( 'background_training_outputs.npy' )
    signal_validation_outputs = np.load( 'signal_validation_outputs.npy' )
    background_validation_outputs = np.load( 'background_validation_outputs.npy' )
    
    plotShapeComparison( signal_training_outputs, background_training_outputs, signal_validation_outputs, background_validation_outputs, 'shapeComparison_overtraining_linear', log = False)
    plotShapeComparison( signal_training_outputs, background_training_outputs, signal_validation_outputs, background_validation_outputs, 'shapeComparison_overtraining_log', log = True)	


def outputName( sample_file ):
    return 'outputs_{}.npy'.format( sample_file.split('/')[-1].split('.')[0] )


def makeSampleArrays( sample_file, is_background):
    pfn_branch_names = [ '_JetConstituentPt', '_JetConstituentEta', '_JetConstituentPhi', '_JetConstituentdxy', '_JetConstituentdz', '_JetConstituentdxyErr', '_JetConstituentdzErr', '_JetConstituentNumberOfHits', '_JetConstituentNumberOfPixelHits', '_JetConstituentCharge', '_JetConstituentPdgId']
    highlevel_branch_names = [ '_JetPt', '_JetEta' ]
    label_branch = '_JetIsFromHNL'
    
    f = uproot.open( sample_file )
    tree = f['HNLtagger_tree']
    
    sample_generator = PFNSampleGenerator( tree, pfn_branch_names, highlevel_branch_names, label_branch, validation_fraction = 0.0, test_fraction = 0.0 )
    model = generateModel()
    model.load_weights( '/user/bvermass/public/PFN/JetTagger/jetTagger_reliso_novtx.h5' )
    
    generator = sample_generator.trainingGenerator( 512 )
    outputs, labels = getOutputArray( generator, sample_generator.numberOfTrainingBatches( 512 ), model )
    #outputs, labels = getOutputArray( generator, 4, model )
    signal_outputs, background_outputs = signalBackgroundSplit( outputs, labels)
    
    output_name = outputName( sample_file )
    if is_background:
    	np.save( output_name, background_outputs )
    else:
    	np.save( output_name, signal_outputs )


def plotProcessComparison( bkg_output_map, signal_output_map, name, log):
    all_outputs = [ bkg_output_map[key] for key in bkg_output_map ]
    for key in signal_output_map:
        all_outputs.append( signal_output_map[key] )
    min_output = min( np.min( output ) for output in all_outputs )
    max_output = max( np.max( output ) for output in all_outputs )
    
    colors = ['blue', 'red', 'orange', 'purple', 'green', 'cyan' ]
    
    for i, key in enumerate( bkg_output_map ):
        addHist( bkg_output_map[key], np.ones( len( bkg_output_map[key] ) ), 30, min_output, max_output, key, color = colors[ i ] )

    j = int( i + 1)
    for i, key in enumerate( signal_output_map ):
        addHist( signal_output_map[key], np.ones( len( signal_output_map[key] ) ), 30, min_output, max_output, key, color = colors[ i + j ] )
        
    
    plt.xlabel( 'Model output', fontsize = 22 )
    plt.ylabel( 'Normalized events', fontsize = 22 )
    if log:
        plt.yscale( 'log' )
    plt.legend(ncol=2, prop={'size': 10})
    
    bottom, top = plt.ylim()
    if log:
        plt.ylim( bottom, top*10 )
    else:
        plt.ylim( 0,  top*1.2)
    plt.savefig(name if '.' in name else name + '.pdf')
    plt.clf() 


def plotProcessRocCurves( bkg_output_map, signal_output_map, name, log ):
	#ROC curves 
	for signal_key in signal_output_map:
		roc_curves = {}
		for bkg_key in bkg_output_map:
		    sig_eff, bkg_eff = computeROC( signal_output_map[ signal_key ], np.ones( len( signal_output_map[ signal_key ] ) ), bkg_output_map[bkg_key], np.ones( len( bkg_output_map[ bkg_key ] ) ), num_points = 1000 )
		    roc_curves[bkg_key] = ( sig_eff, bkg_eff )
		
		colors = ['blue', 'red', 'orange', 'purple', 'green', 'cyan' ]
		for i, bkg_key in enumerate( bkg_output_map ):
		    plt.plot( roc_curves[bkg_key][0], backgroundRejection( roc_curves[bkg_key][1] ), colors[i], lw =2, label = bkg_key )
		
		plt.xlabel( 'Signal efficiency', fontsize = 18 )
		plt.ylabel( 'Background rejection', fontsize = 18 )
		plt.grid(True)
		plt.legend( loc = 'best', prop={'size': 15})
		if log:
			plt.yscale( 'log' )
			plt.xscale( 'log' )

		plt.savefig(name + signal_key if '.' in name else name + signal_key + '.pdf')
		
		#clear canvas
		plt.clf()
 


if __name__ == '__main__':
    makeOutputArrays()
    producePlots()
    
    bkg_samples = {'Drell-Yan' : [ '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_Background_DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.root',  '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_Background_DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.root'],
        'W + jets' : [ '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_Background_WJetsToLNu_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root' ],
        'TT semilep.' : ['/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_Background_TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root', '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_Background_TTJets_SingleLeptFromTbar_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root' ],
        'TT dilep.' : ['/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_Background_TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root'] 
    }
    signal_samples = {'HNL' : ['/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_HeavyNeutrino_lljj_M-5_mu.root'] }
    
    bkg_output_map = {}
    
    for process in bkg_samples:
        for sample in bkg_samples[process]:
            makeSampleArrays( sample, True)
    
    for process in bkg_samples:
        for sample in bkg_samples[process]:
            temp = np.load( outputName( sample ) )
            if process in bkg_output_map:
                bkg_output_map[ process ] = np.concatenate( [bkg_output_map[ process ], temp ], axis = -1 )
            else:
                bkg_output_map[ process ] = temp
    
    
    signal_output_map = {}
    
    for process in signal_samples:
        for sample in signal_samples[process]:
            makeSampleArrays( sample, False)
    
    for process in signal_samples:
        for sample in signal_samples[process]:
            temp = np.load( outputName( sample ) )
            if process in signal_output_map:
                signal_output_map[ process ] = np.concatenate( [signal_output_map[ process ], temp ], axis = -1 )
            else:
                signal_output_map[ process ] = temp 
    plotProcessComparison( bkg_output_map,  signal_output_map, 'processComparison', log = False)
    plotProcessComparison( bkg_output_map,  signal_output_map, 'processComparison_log', log = True)
    
    plotProcessRocCurves( bkg_output_map,  signal_output_map, 'processROC', log = False)
    plotProcessRocCurves( bkg_output_map,  signal_output_map, 'processROC_log', log = True)
    """
    #ROC curves 
    for signal_key in signal_output_map:
        roc_curves = {}
        for bkg_key in bkg_output_map:
            sig_eff, bkg_eff = computeROC( signal_output_map[ signal_key ], np.ones( len( signal_output_map[ signal_key ] ) ), bkg_output_map[bkg_key], np.ones( len( bkg_output_map[ bkg_key ] ) ), num_points = 1000 )
            roc_curves[bkg_key] = ( sig_eff, bkg_eff )
    
        colors = ['blue', 'red', 'orange', 'purple', 'green', 'cyan' ]
        for i, bkg_key in enumerate( bkg_output_map ):
            plt.plot( roc_curves[bkg_key][0], backgroundRejection( roc_curves[bkg_key][1] ), colors[i], lw =2, label = bkg_key )
            
        plt.xlabel( 'Signal efficiency', fontsize = 18 )
        plt.ylabel( 'Background rejection', fontsize = 18 )
        plt.grid(True)
        plt.legend( loc = 'best', prop={'size': 15})
        plt.savefig('roc.pdf')
        
        #clear canvas
        plt.clf()
    """
