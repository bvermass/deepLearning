import os
import sys
import subprocess

#import deepsets stuff

#import jobsubmission script
from configuration.LearningAlgorithms import *
from configuration.InputReader import *
from jobSubmission.submitJob import submitProcessJob

def submitTrainingPFNJob( configuration, number_of_threads, high_memory, root_file_name, output_directory ):

    #model name 
    model_name = configuration.name() 
    print( 'model name: {}'.format( model_name ) )
    print( 'output directory name: {}'.format( output_directory ) )
    
    #new directory for job output
    os.system('mkdir -p {}/{}'.format( output_directory, model_name ) )

    #make the command to run, starting with switching to the output directory
    command_string = 'cd {}/{}\n'.format( output_directory, model_name )

    #add the training to the command
    main_directory = os.path.dirname( os.path.abspath( __file__ ) )
    command_string += 'python {}/DeepSets/trainPFN.py {}'.format( main_directory, root_file_name )
    for name, value in configuration:
        command_string += ' {}={}'.format( name, value )
    
    #pipe output to text files 
    log_file = 'trainingOutput_' + model_name + '.txt'
    error_file = model_name + '_err.txt'
    command_string += ' > {} 2>{} '.format( log_file, error_file)
    print( command_string )
    
    #dump configuration to output directory 
    configuration.toJSON( os.path.join( output_directory, model_name, 'configuration_' + model_name + '.json' ) )
    
    #submit this process 
    return submitProcessJob( command_string, 'trainPFNModel.sh', wall_time = '24:00:00', num_threads = number_of_threads, high_memory = high_memory )

def submitTrainingPFNJobs( configuration_file_name ):

    configuration_file = __import__( configuration_file_name.replace('.py', '') )
   
    #root input file
    if hasattr( configuration_file, 'root_file_name' ):
        root_file_name = configuration_file.root_file_name
    else:
        print( 'Error : No root file name specified in input file' )
        print( 'Please add root_file_name to input file' )
        sys.exit()

    #directory for job output
    output_directory_name = '/user/bvermass/public/PFN/JetTagger/{}'.format( configuration_file_name.replace('input_', '').replace('.py', '') )

    #Make list of PFN network configurations to process
    grid_scan_configuration = GridScanInputReader( configuration_file )
    configuration_list = [ config for config in grid_scan_configuration ]
    
    number_of_models = len( grid_scan_configuration )
    max_number_of_trainings = 2500
    if number_of_models > max_number_of_trainings :
        print( 'Error : requesting to train {} models. The cluster only allows {} jobs to be submitted.'.format( number_of_models, max_number_of_trainings ) )
        print( 'Please modify the configuration file to train less models.')
        sys.exit()
    else :
        print ( 'submitting {} models'.format( number_of_models ) )

    #check if any job submission requirements were specified 
    number_of_threads = 1 
    if hasattr( configuration_file, 'number_of_threads' ):
        number_of_threads = configuration_file.number_of_threads
    high_memory = False
    if hasattr( configuration_file, 'high_memory' ):
        high_memory = configuration_file.high_memory

    #submitting a job for each configuration
    job_id_list = []
    for configuration in configuration_list:
        job_id = submitTrainingPFNJob( configuration, number_of_threads, high_memory, root_file_name, output_directory_name )
        job_id_list.append( job_id )
    
    print( '########################################################' )
    print( 'Submitted {} neural networks for training.'.format( number_of_models ) )
    print( '########################################################' )

def trainPFNModel():
    #main_directory = os.path.dirname( os.path.abspath( __file__ ) )
    
    #check if atleast one additional argument is given, the program expects a configuration file to run
    if len( sys.argv ) < 2:
        print( 'Error: incorrect number of arguments given to script.')
        print( 'Usage: <python runTraining.py configuration.py>')
        sys.exit()
    
    #read configuration file 
    configuration_file_name = sys.argv[1]
    
    #train PFN models from configuration
    submitTrainingPFNJobs( configuration_file_name )
    
    
    #print('python {}/DeepSets/trainPFN.py {}'.format(main_directory, inputfile))
    #submitProcessJob('python {}/DeepSets/trainPFN.py {}'.format(main_directory, inputfile), '{}/trainPFNscript_'.format(main_directory), wall_time='24:00:00')

if __name__ == '__main__' :
    trainPFNModel()
