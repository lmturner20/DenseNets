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

def find_top_ligand(results):
	numTargets = results.length
	currentTarget = ("")
	highestLigands = [numTargets]
	index = 0
	correctPoses = 0
	highestPose = 0
	rightAnswer = false

	for r in results:
		if r[2] = currentTarget:
			if r[5] > highestPose:
				highestPose = r[5]
				highestLigand = r[3]
				if r[4] = 1:
					rightAnswer = true
				else:
					rightAnswer = false
		else:
			currentTarget = r[2];
			highestLigands[index] = highestLigand;
			index = index + 1
		if rightAnswer = true:
			correctPoses = correctPoses + 1
	print ("For top scoring ligands: percent of correct poses = " + correctPoses/numTargets + "\n")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',type=str,required=True)
    parser.add_argument('-p','--prefix',type=str,required=True) #same for caffemodel and test sets
    args = parser.parse_args()
    modelname = (args.model + ".model")

    testfile = (args.prefix + "train0.types")
    caffemodel = (args.model + ".0_iter_100000.caffemodel")
    #results = evaluate_fold(testfile, caffemodel, modelname)
    results = [ '8a3h/8a3h_ligand3_22.gninatypes', 0.0, 0.07919405400753021), (-4.055500030517578, 4.712827205657959, '8a3h/8a3h_rec.gninatypes', '8a3h/8a3h_ligand3_24.gninatypes', 0.0, 0.34352347254753113), (7.638299942016602, 6.938665390014648, '966c/966c_rec.gninatypes', '966c/966c_ligand3_0.gninatypes', 1.0, 0.9758247137069702), (-7.638299942016602, 4.961941242218018, '966c/966c_rec.gninatypes', '966c/966c_ligand3_1.gninatypes', 0.0, 0.02706882543861866), (-7.638299942016602, 5.282212257385254, '966c/966c_rec.gninatypes', '966c/966c_ligand3_2.gninatypes', 0.0, 0.04771214351058006), (-7.638299942016602, 5.643853187561035, '966c/966c_rec.gninatypes', '966c/966c_ligand3_5.gninatypes', 0.0, 0.15937583148479462), (-7.638299942016602, 6.056888580322266, '966c/966c_rec.gninatypes', '966c/966c_ligand3_11.gninatypes', 0.0, 0.7478074431419373), (-7.638299942016602, 5.352646827697754, '966c/966c_rec.gninatypes', '966c/966c_ligand3_13.gninatypes', 0.0, 0.08677862584590912), (-7.638299942016602, 6.0087738037109375, '966c/966c_rec.gninatypes', '966c/966c_ligand3_14.gninatypes', 0.0, 0.8416995406150818), (-7.638299942016602, 4.911555290222168, '966c/966c_rec.gninatypes', '966c/966c_ligand3_16.gninatypes', 0.0, 0.373503714799881), (-7.638299942016602, 5.491366863250732, '966c/966c_rec.gninatypes', '966c/966c_ligand3_17.gninatypes', 0.0, 0.36361685395240784), (-7.638299942016602, 6.7745819091796875, '966c/966c_rec.gninatypes', '966c/966c_ligand3_18.gninatypes', 0.0, 0.9699649214744568), (-7.638299942016602, 5.881802558898926, '966c/966c_rec.gninatypes', '966c/966c_ligand3_19.gninatypes', 0.0, 0.6239776015281677), (-7.638299942016602, 5.554421901702881, '966c/966c_rec.gninatypes', '966c/966c_ligand3_20.gninatypes', 0.0, 0.39980071783065796), (-7.638299942016602, 5.463313579559326, '966c/966c_rec.gninatypes', '966c/966c_ligand3_21.gninatypes', 0.0, 0.6280282139778137), (-7.638299942016602, 5.830012321472168, '966c/966c_rec.gninatypes', '966c/966c_ligand3_22.gninatypes', 0.0, 0.6617482304573059), (-7.638299942016602, 5.905783176422119, '966c/966c_rec.gninatypes', '966c/966c_ligand3_23.gninatypes', 0.0, 0.5545428395271301)]
    find_top_ligand(results)
    #print (results)
    #print analyze_results(results, args.prefix + ".results", "affinity")
    
    #testfile = (args.prefix + "train1.types")
    #caffemodel = (args.model + ".1_iter_100000.caffemodel")
    #result = evaluate_fold(testfile, caffemodel, modelname)
    #print (result)

    #testfile = (args.prefix + "train2.types")
    #caffemodel = (args.model + ".2_iter_100000.caffemodel")
    #result = evaluate_fold(testfile, caffemodel, modelname)
    #print (result)
