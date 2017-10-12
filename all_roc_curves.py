import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn, argparse
import sklearn.metrics

2npncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_nopool_noconcat.auc.finaltest"
2npwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_nopool_wconcat.auc.finaltest"
2wpncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_wpool_noconcat.auc.finaltest"
2wpwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_wpool_wconcat.auc.finaltest"
3npncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_nopool_noconcat.auc.finaltest"
#3npwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_nopool_wconcat.auc.finaltest"
#3wpncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_wpool_noconcat.auc.finaltest"
3wpwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_wpool_wconcat.auc.finaltest"
outprefix = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/8_roc_curves"

2npnctrue = [] #dn_2mod_nopool_noconcat
2npncscore = []
2npncfile = open(2npncf,'r')
for line in 2npncfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        2npnctrue.append( bool(float(data[0])) )
        2npncscore.append( float(data[1].strip()) )
2npncfpr, 2npnctpr, _ = sklearn.metrics.roc_curve(2npnctrue,2npncscore)
2npnc = sklearn.metrics.roc_auc_score(2npnctrue,2npncscore)

2npwctrue = [] #dn_2mod_nopool_wconcat
2npwcscore = []
2npwcfile = open(2npwcf,'r')
for line in 2npwcfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        2npwctrue.append( bool(float(data[0])) )
        2npwcscore.append( float(data[1].strip()) )
2npwcfpr, 2npwctpr, _ = sklearn.metrics.roc_curve(2npwctrue,2npwcscore)
2npwc = sklearn.metrics.roc_auc_score(2npwctrue,2npwcscore)

2wpnctrue = [] #dn_2mod_wpool_noconcat
2wpncscore = []
2wpncfile = open(2wpncf,'r')
for line in 2wpncfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        2wpnctrue.append( bool(float(data[0])) )
        2wpncscore.append( float(data[1].strip()) )
2wpncfpr, 2wpnctpr, _ = sklearn.metrics.roc_curve(2wpnctrue,2wpncscore)
2wpnc = sklearn.metrics.roc_auc_score(2wpnctrue,2wpncscore)

2wpwctrue = [] #dn_2mod_wpool_wconcat
2wpwcscore = []
2wpwcfile = open(2wpwcf,'r')
for line in 2wpwcfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        2wpwctrue.append( bool(float(data[0])) )
        2wpwcscore.append( float(data[1].strip()) )
2wpwcfpr, 2wpwctpr, _ = sklearn.metrics.roc_curve(2wpwctrue,2wpwcscore)
2wpwc = sklearn.metrics.roc_auc_score(2wpwctrue,2wpwcscore)

3npnctrue = [] #dn_3mod_nopool_noconcat
3npncscore = []
3npncfile = open(3npncf,'r')
for line in 3npncfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        3npnctrue.append( bool(float(data[0])) )
        3npncscore.append( float(data[1].strip()) )
3npncfpr, 3npnctpr, _ = sklearn.metrics.roc_curve(3npnctrue,3npncscore)
3npnc = sklearn.metrics.roc_auc_score(3npnctrue,3npncscore)

#3npwctrue = [] #dn_3mod_nopool_wconcat
#3npwcscore = []
#3npwcfile = open(3npwcf,'r')
#for line in 3npwcfile:
#    if line.startswith('#'):
#        pass
#    else:
#        data= line.split()
#        3npwctrue.append( bool(float(data[0])) )
#        3npwcscore.append( float(data[1].strip()) )
#3npwcfpr, 3npwctpr, _ = sklearn.metrics.roc_curve(3npwctrue,3npwcscore)
#3npwc = sklearn.metrics.roc_auc_score(3npwctrue,3npwcscore)

#3wpnctrue = [] #dn_3mod_wpool_noconcat3
#3wpncscore = []
#3wpncfile = open(3wpncf,'r')
#for line in 3wpncfile:
#    if line.startswith('#'):
#        pass
#    else:
#        data= line.split()
#        3wpnctrue.append( bool(float(data[0])) )
#        3wpncscore.append( float(data[1].strip()) )
#3wpncfpr, 3wpnctpr, _ = sklearn.metrics.roc_curve(3wpnctrue,3wpncscore)
#3wpnc = sklearn.metrics.roc_auc_score(3wpnctrue,3wpncscore)

3wpwctrue = [] #dn_3mod_wpool_wconcat
3wpwcscore = []
3wpwcfile = open(3wpwcf,'r')
for line in 3wpwcfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        3wpwctrue.append( bool(float(data[0])) )
        3wpwcscore.append( float(data[1].strip()) )
3wpwcfpr, 3wpwctpr, _ = sklearn.metrics.roc_curve(3wpwctrue,3wpwcscore)
3wpwc = sklearn.metrics.roc_auc_score(3wpwctrue,3wpwcscore)


fig = plt.figure(figsize=(8,8))
plt.plot(2npncfpr,2npnctpr,label='2NPNC (AUC=%.4f)'%(2npnc),linewidth=8,color='b')
plt.plot(2npwcfpr,2npwctpr,label='2NPWC (AUC=%.4f)'%(2npwc),linewidth=8,color='g')
plt.plot(2wpncfpr,2wpnctpr,label='2WPNC (AUC=%.4f)'%(2wpnc),linewidth=8,color='r')
plt.plot(2wpwcfpr,2wpwctpr,label='2WPWC (AUC=%.4f)'%(2wpwc),linewidth=8,color='c')

plt.plot(3npncfpr,3npnctpr,label='3NPNC (AUC=%.4f)'%(3npnc),linewidth=8,color='m')
#plt.plot(3npwcfpr,3npwctpr,label='3NPWC (AUC=%.4f)'%(3npwc),linewidth=8,color='y')
#plt.plot(3wpncfpr,3wpnctpr,label='3WPNC (AUC=%.4f)'%(3wpnc),linewidth=8,color='k')
plt.plot(3wpwcfpr,3wpwctpr,label='3WPWC (AUC=%.4f)'%(3wpwc),linewidth=8,color='w')
plt.legend(loc='lower right',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=22)
plt.ylabel('True Positive Rate',fontsize=22)
plt.axes().set_aspect('equal')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('%s_roc.pdf'%outprefix,bbox_inches='tight')
