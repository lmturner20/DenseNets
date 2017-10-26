from Bio.PDB.PDBParser import PDBParser
import numpy as np
import os, sys
os.environ["GLOG_minloglevel"] = "10"
sys.path.append("/home/dkoes/git/gninascripts/")
sys.path.append("/net/pulsar/home/koes/dkoes/git/gninascripts/")

import train, predict
import matplotlib, caffe
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys
import sklearn.metrics
import scipy.stats

def evaluate_fold(testfile, caffemodel, modelname):
    '''Evaluate the passed model and the specified test set.
    Assumes the .model file is named a certain way.
    Returns tuple:
    (correct, prediction, receptor, ligand, label (optional), posescore (optional))
    label and posescore are only provided is trained on pose data
    ''' 
    print ("0")
    caffe.set_mode_gpu()
    print ("0.1")
    test_model = 'predict.%d.prototxt' % os.getpid()
    print ("0.2")
    print ("test_model:" + test_model)
    train.write_model_file(test_model, modelname, testfile, testfile, '')
    print ("0.3")
    test_net = caffe.Net(test_model, caffemodel, caffe.TEST)
    print ("0.4")
    lines = open(testfile).readlines()
    print ("1")

    res = None
    i = 0 #index in batch
    correct = 0
    prediction = 0
    receptor = ''
    ligand = ''
    label = 0
    posescore = -1
    ret = []
    print ("2")
    for line in lines:
        #check if we need a new batch of results
        if not res or i >= batch_size:
            res = test_net.forward()
            if 'output' in res:
                batch_size = res['output'].shape[0]
            else:
                batch_size = res['affout'].shape[0]
            i = 0
            print ("3")

        if 'labelout' in res:
            label = float(res['labelout'][i])
        if 'output' in res:
            posescore = float(res['output'][i][1])
        if 'affout' in res:
            correct = float(res['affout'][i])
        if 'predaff' in res:
            prediction = float(res['predaff'][i])
            if not np.isfinite(prediction).all():
                os.remove(test_model)
                return [] #gracefully handle nan?
        print ("4")

        #extract ligand/receptor for input file
        tokens = line.split()
        for t in xrange(len(tokens)):
            if tokens[t].endswith('gninatypes'):
                receptor = tokens[t]
                ligand = tokens[t+1]
                break
        print ("5")
        
        #(correct, prediction, receptor, ligand, label (optional), posescore (optional))       
        if posescore < 0:
            ret.append((correct, prediction, receptor, ligand))
        else:
            ret.append((correct, prediction, receptor, ligand, label, posescore))
            
        i += 1 #batch index
        print ("6")
        
    os.remove(test_model)
    print ("7")
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',type=str,required=True)
    parser.add_argument('-p','--prefix',type=str,required=True) #same for caffemodel and test sets
    args = parser.parse_args()
    modelname = args.model

    testfile = (args.prefix + ".0.rmsd.finaltest")
    caffemodel = (args.prefix + ".0_iter_100000.caffemodel")
    print ("train:" + train.__file__)
    print ("testfile:" + testfile)
    print ("caffemodel:" + caffemodel)
    print ("modelname:" + modelname)
    result = evaluate_fold(testfile, caffemodel, modelname)
    print ("evaluate_fold returned")
    print (result)

    testfile = (prefix + ".1.rmsd.finaltest")
    caffemodel = (prefix + ".1_iter_100000.caffemodel")
    result = evaluate_fold(testfile, caffemodel, modelname)
    print (result)

    testfile = (prefix + ".2.rmsd.finaltest")
    caffemodel = (prefix + ".2_iter_100000.caffemodel")
    result = evaluate_fold(testfile, caffemodel, modelname)
    print (result)
