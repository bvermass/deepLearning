import os
import sys
import subprocess

#import deepsets stuff

#import jobsubmission script
from jobSubmission.submitJob import submitProcessJob

def trainPFNModel():
    main_directory = os.path.dirname( os.path.abspath( __file__ ) )
    inputfile = '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/mergedFile_randomized.root'
    print('python {}/DeepSets/trainPFN.py {}'.format(main_directory, inputfile))
    submitProcessJob('python {}/DeepSets/trainPFN.py {}'.format(main_directory, inputfile), '{}/trainPFNscript_'.format(main_directory), wall_time='24:00:00')

if __name__ == '__main__' :
    trainPFNModel()
