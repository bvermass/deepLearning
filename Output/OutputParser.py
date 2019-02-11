import os
import operator
import sys
import json
from collections import OrderedDict 

#import other parts of framework
sys.path.insert(0, '../configuration')
sys.path.insert(0, '../')
from stringTools import removeLeadingCharacter, canConvertToFloat
from Configuration import *
from InputReader import *


def listContent( directory_name, typeCheck):
    for entry in os.listdir( directory_name ):
        full_path = os.path.join( directory_name, entry )
        if typeCheck( full_path ) :
            yield full_path


def listSubDirectories( directory_name ):
    return listContent( directory_name, os.path.isdir )


def listFiles( directory_name ):
    return listContent( directory_name, os.path.isfile )


#extract model name from configuration or training output file name 
def extractModelName( file_name ):
    prefix = file_name.split('/')[-1].split('_')[0]
    return file_name.split('.')[0].split(prefix + '_')[-1]


class OutputParser:

    def __init__( self, output_directory_name ):

        self._output_directory_name = output_directory_name
        self._AUC_map = OrderedDict()

        #each model is assumed to have a file with the training output (named trainingOutput_ followed by the model name), and a file with the configuration ( named configuration_ followed by the model name )
        output_files = [ os.path.join( subdirectory, f ) for subdirectory, _, files in os.walk( self._output_directory_name ) for f in files ]

        configuration_files = [ f for f in output_files if 'configuration_' in f  ]
        training_output_files = [ f for f in output_files if 'trainingOutput_' in f ]

        file_pairs = ( (train, config) for config in configuration_files for train in training_output_files if extractModelName( train ) == extractModelName( config ) )
        
        for train, config in file_pairs:

            with open( train ) as trainingOutput:
                AUC_line_counter = 0
                for line in trainingOutput.readlines():
				
                    #expect the output file to contain one line in the following format: "validation set ROC integral (AUC) = X" 
                    #make the code slightly more robust for potential changes in the output format
                    if ('AUC' in line) or ('ROC integral' in line ):
                        AUC_line_counter += 1
                        numbers_in_line = [ float(s) for s in line.split() if canConvertToFloat(s) ]
                        if len( numbers_in_line ) != 1 or AUC_line_counter > 1:
                            raise IndexError('Expect to find exactly 1 number representing the ROC integral in {}, but found {}.'.format(trainingOutput_file_name, len( numbers_in_line) ) )
                        AUC = numbers_in_line[0]

            #read model configuration from configuration file 
            model_configuration = newConfigurationFromJSON( config )
            
            #fill entry in in map collecting training information 
            self._AUC_map[model_configuration] = AUC


    def rankModels(self):
        self.AUC_map = OrderedDict( sorted( self._AUC_map.items(), key = operator.itemgetter(1), reverse=True ) )


    def printBestModels(self):
        for i, model in enumerate( self._AUC_map.items() ):
            if i >= 10:
                break
            print( '########################################################') 
            print( 'Rank {}:'.format( i + 1 ) ) 
            print( model[0].name() )
            print( 'validation set ROC integral (AUC) = {}'.format( model[1] ) )

    
    def outputName(self):
        output_name = self._output_directory_name
        output_name = output_name.replace( 'output', '' )
        output_name = removeLeadingCharacter( output_name, '_' )
        return output_name

    
    def copyBestModelsOutput(self):
        best_model_directory = 'bestModels_{}'.format( self.outputName() )
        os.system('mkdir -p {}'.format( best_model_directory ) )
        for i, model in enumerate( self._AUC_map.items() ):
            if i >= 10:
                break
            os.system('cp -r {0}/{1} {2}/model_rank_{3}'.format( self.output_directory, model[0].name(), best_model_directory, i + 1 ) )


    def bestModels(self):
        self.rankModels()
        self.copyBestModelsOutput()
        self.printBestModels()

    
    def toGeneration(self, input_reader):
        config_generator = ( config for config in self._AUC_map.keys() )
        return neuralNetworkConfigurationsAndInputToGeneration( config_generator , input_reader )


    def getAUC( self, config ):
        return self._AUC_map[config]

        

if __name__ == '__main__' :
    
    if len( sys.argv ) == 2 :
        output_directory = sys.argv[1]
        ranker = OutputParser( output_directory )
        ranker.bestModels()
    else :
        print( 'Error: incorrect number of arguments given to script.')
        print( 'Usage: <python OutputParser.py output_directory>')
     
