from Bio.PDB.PDBParser import PDBParser
import numpy as np
import os, sys
import os.path
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
    num_targets=len(targets)
    correct_poses=0
    ligands=[]

    for r in results:
        if r[2] in targets.keys():
            targets[r[2]].append((r[5], r[4], r[3]))#also ligand
        else:
            targets[r[2]] = [(r[5], r[4], r[3])]

    for t in targets.keys():
        targets[t].sort
        top_tuples = targets[t][0:topnum]
        for i in top_tuples:
            if i[1] == 1:
                correct_poses = correct_poses + 1
                ligands.append(i[2])
                break

    percent = correct_poses/num_targets*100.0
    return (percent, ligands)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',type=str,required=True,help='Model filename')
    parser.add_argument('-p','--prefix',type=str,required=True,help='Prefix for test files')
    parser.add_argument('-c','--caffemodel',type=str,required=True,help='Prefix for caffemodel file')
    parser.add_argument('-o','--output',type=str,required=True,help='Output filename')
    parser.add_argument('-f','--folds',type=int,default=3,help='Number of folds')
    parser.add_argument('-i','--iterations',type=int,default=0,help='Iterations in caffemodel filename')
    parser.add_argument('-t','--top',type=int,default=0,help='Number of top ligands to look at')
    args = parser.parse_args()

    iterations=args.iterations
    if iterations == 0:
        highest_iter=0
        for name in glob.glob('dir/*.caffemodel'):
            new_iter=int(filter(str.isdigit, name))
            if new_iter>highest_iter:
                highest_iter=new_iter
        iterations=highest_iter
        
    modelname = (args.model)
    testfile = (args.prefix + "train0.types")
    output = (args.output)

    results=[]
    for f in range(args.folds):
        caffemodel=(args.caffemodel+'.'+str(f)+'_iter_'+str(iterations)+'.caffemodel')
        if (os.path.isfile(caffemodel) == False):
            print ('Error: Caffemodel file does not exist. Check --caffemodel, --iterations, and --folds arguments.')
        results += evaluate_fold(testfile, caffemodel, modelname)
    
    top = find_top_ligand(results,1)
    file=open(output, "w")
    write("Percent of targets that contain the correct pose in the top 1:"+str(top[0])+"%")
    write("Top ligands: ")
    for l in top[1]:
          write(l)
    file.close()
          
    find_top_ligand(results,3)
    file=open(output, "a")
    write("Percent of targets that contain the correct pose in the top 3:"+str(top[0])+"%")
    write("Top ligands: ")
    for l in top[1]:
          write(l)
    file.close()
          
    find_top_ligand(results,5)
    file=open(output, "a")
    write("Percent of targets that contain the correct pose in the top 5:"+str(top[0])+"%")
    write("Top ligands: ")
    for l in top[1]:
          write(l)
    file.close()
          
    if args.top > 0:
        find_top_ligand(results,args.top)
        file=open(output, "a")
        write("Percent of targets that contain the correct pose in the top"+args.top+":"+str(top[0])+"%")
        write("Top ligands: ")
        for l in top[1]:
            write(l)
        file.close()
