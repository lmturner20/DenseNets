from Bio.PDB.PDBParser import PDBParser
import numpy as np
import os, sys
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
    caffe.set_mode_gpu()
    test_model = ('predict.%d.prototxt' % os.getpid())
    print ("test_model:" + test_model)
    train.write_model_file(test_model, modelname, testfile, testfile, '')
    test_net = caffe.Net(test_model, caffemodel, caffe.TEST)
    lines = open(testfile).readlines()

    res = None
    i = 0 #index in batch
    correct = 0
    prediction = 0
    receptor = ''
    ligand = ''
    label = 0
    posescore = -1
    ret = []
    
    for line in lines:
        #check if we need a new batch of results
        if not res or i >= batch_size:
            res = test_net.forward()
            if 'output' in res:
                batch_size = res['output'].shape[0]
            else:
                batch_size = res['affout'].shape[0]
            i = 0

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

        #extract ligand/receptor for input file
        tokens = line.split()
        for t in xrange(len(tokens)):
            if tokens[t].endswith('gninatypes'):
                receptor = tokens[t]
                ligand = tokens[t+1]
                break
        
        #(correct, prediction, receptor, ligand, label (optional), posescore (optional))       
        if posescore < 0:
            ret.append((correct, prediction, receptor, ligand))
        else:
            ret.append((correct, prediction, receptor, ligand, label, posescore))
            
        i += 1 #batch index
        
    os.remove(test_model)
    return ret

def find_top_ligand(results, topnum):
    targets={}
    num_targets = 0
    correct_poses = 0

    for r in results:
        if r[2] in targets.keys():
            targets[r[2]].append((r[5], r[4]))
        else:
            targets[r[2]] = [(r[5], r[4])]

    for t in targets.keys():
        num_targets = num_targets + 1
        targets[t].sort
        top_tuples = targets[t][0:topnum]
        for i in top_tuples:
            if i[1] == 1:
                correct_poses = correct_poses + 1
                break

    percent = correct_poses/num_targets*100
    print ("Percent of targets that contain the correct pose in the top "+str(topnum)+": "+str(percent)+"%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',type=str,required=True)
    parser.add_argument('-p','--prefix',type=str,required=True) #same for caffemodel and test sets
    args = parser.parse_args()
    modelname = (args.model + ".model")

    testfile = (args.prefix + "train0.types")
    caffemodel = (args.model + ".0_iter_100000.caffemodel")
    results = evaluate_fold(testfile, caffemodel, modelname)
    print (args.prefix)
    print("0:")
    find_top_ligand(results,1)
    find_top_ligand(results,3)
    find_top_ligand(results,5)

    testfile = (args.prefix + "train1.types")
    caffemodel = (args.model + ".1_iter_100000.caffemodel")
    results = evaluate_fold(testfile, caffemodel, modelname)
    print("1:")
    find_top_ligand(results,1)
    find_top_ligand(results,3)
    find_top_ligand(results,5)

    testfile = (args.prefix + "train2.types")
    caffemodel = (args.model + ".2_iter_100000.caffemodel")
    results = evaluate_fold(testfile, caffemodel, modelname)
    print("2:")
    find_top_ligand(results,1)
    find_top_ligand(results,3)
    find_top_ligand(results,5)

