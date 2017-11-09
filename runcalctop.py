#!/bin/bash
#PBS -N afftrain
#PBS -j oe
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=24:00:00:00
#PBS -q dept_gpu

echo Running on `hostname`
echo workdir $PBS_O_WORKDIR
echo ld_library_path $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/local/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-8.0/lib64:/usr/lib
export PYTHONPATH=/net/pulsar/home/koes/dkoes/local/python
export PATH=$PATH:$HOME/git/gninascripts

#for some reason when the job starts on the node, it auto-cd's to your home dir
cd $PBS_O_WORKDIR

cmd=`sed -n "${PBS_ARRAYID}p" cmds`
echo $cmd
$cmd 

exit
