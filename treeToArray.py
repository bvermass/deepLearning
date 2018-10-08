import root_numpy
from ROOT import TFile 
import numpy as np

def treeToArray(tree, branchList, cut = ''):

    #convert tree to array of tuples 
    arrayTuples = root_numpy.tree2array(tree, branchList, cut)

    #convert to ndarray
    output_shape = ()
    num_rows = len(arrayTuples)

    #check if one branch name or list of branches is given
    argument_is_list = not ( type(branchList) is str ) 

    if argument_is_list :
        num_columns = len(arrayTuples[0])
        output_shape = (num_rows, num_columns)
    else :
        output_shape = (num_rows, )

    retArray = np.zeros( (output_shape) )
    for i, entry in enumerate(arrayTuples):
        if argument_is_list:
            entry = list(entry)
            retArray[i] = np.asarray( entry )
        else :
            retArray[i] = entry

    return retArray
    
def writeArrayToFile(array, fileName):
    np.save( fileName , array)

def loadArray(fileName):
    return np.load(fileName)

def listOfBranches(tree):
    names = [ branch.GetName() for branch in tree.GetListOfBranches() ]
    return names 

if __name__ == '__main__' :
    pass
    
