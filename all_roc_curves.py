import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn, argparse
import sklearn.metrics

_2npncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_nopool_noconcat.auc.finaltest"
_2npwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_nopool_wconcat.auc.finaltest"
_2wpncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_wpool_noconcat.auc.finaltest"
_2wpwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_2mod_wpool_wconcat.auc.finaltest"
_3npncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_nopool_noconcat.auc.finaltest"
_3npwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_nopool_wconcat.auc.finaltest"
_3wpncf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_wpool_noconcat.auc.finaltest"
_3wpwcf = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/dn_3mod_wpool_wconcat.auc.finaltest"
outprefix = "/net/pulsar/home/koes/lmt72/DenseNets/DenseNets/8_roc_curves"

_2npnctrue = [] #dn_2mod_nopool_noconcat
_2npncscore = []
_2npncfile = open(_2npncf,'r')
for line in _2npncfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _2npnctrue.append( bool(float(data[0])) )
        _2npncscore.append( float(data[1].strip()) )
_2npncfpr, _2npnctpr, _ = sklearn.metrics.roc_curve(_2npnctrue,_2npncscore)
_2npnc = sklearn.metrics.roc_auc_score(_2npnctrue,_2npncscore)

_2npwctrue = [] #dn_2mod_nopool_wconcat
_2npwcscore = []
_2npwcfile = open(_2npwcf,'r')
for line in _2npwcfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _2npwctrue.append( bool(float(data[0])) )
        _2npwcscore.append( float(data[1].strip()) )
_2npwcfpr, _2npwctpr, _ = sklearn.metrics.roc_curve(_2npwctrue,_2npwcscore)
_2npwc = sklearn.metrics.roc_auc_score(_2npwctrue,_2npwcscore)

_2wpnctrue = [] #dn_2mod_wpool_noconcat
_2wpncscore = []
_2wpncfile = open(_2wpncf,'r')
for line in _2wpncfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _2wpnctrue.append( bool(float(data[0])) )
        _2wpncscore.append( float(data[1].strip()) )
_2wpncfpr, _2wpnctpr, _ = sklearn.metrics.roc_curve(_2wpnctrue,_2wpncscore)
_2wpnc = sklearn.metrics.roc_auc_score(_2wpnctrue,_2wpncscore)

_2wpwctrue = [] #dn_2mod_wpool_wconcat
_2wpwcscore = []
_2wpwcfile = open(_2wpwcf,'r')
for line in _2wpwcfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _2wpwctrue.append( bool(float(data[0])) )
        _2wpwcscore.append( float(data[1].strip()) )
_2wpwcfpr, _2wpwctpr, _ = sklearn.metrics.roc_curve(_2wpwctrue,_2wpwcscore)
_2wpwc = sklearn.metrics.roc_auc_score(_2wpwctrue,_2wpwcscore)

_3npnctrue = [] #dn_3mod_nopool_noconcat
_3npncscore = []
_3npncfile = open(_3npncf,'r')
for line in _3npncfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _3npnctrue.append( bool(float(data[0])) )
        _3npncscore.append( float(data[1].strip()) )
_3npncfpr, _3npnctpr, _ = sklearn.metrics.roc_curve(_3npnctrue,_3npncscore)
_3npnc = sklearn.metrics.roc_auc_score(_3npnctrue,_3npncscore)

_3npwctrue = [] #dn_3mod_nopool_wconcat
_3npwcscore = []
_3npwcfile = open(_3npwcf,'r')
for line in _3npwcfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _3npwctrue.append( bool(float(data[0])) )
        _3npwcscore.append( float(data[1].strip()) )
_3npwcfpr, _3npwctpr, _ = sklearn.metrics.roc_curve(_3npwctrue,_3npwcscore)
_3npwc = sklearn.metrics.roc_auc_score(_3npwctrue,_3npwcscore)

_3wpnctrue = [] #dn_3mod_wpool_noconcat3
_3wpncscore = []
_3wpncfile = open(_3wpncf,'r')
for line in _3wpncfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _3wpnctrue.append( bool(float(data[0])) )
        _3wpncscore.append( float(data[1].strip()) )
_3wpncfpr, _3wpnctpr, _ = sklearn.metrics.roc_curve(_3wpnctrue,_3wpncscore)
_3wpnc = sklearn.metrics.roc_auc_score(_3wpnctrue,_3wpncscore)

_3wpwctrue = [] #dn_3mod_wpool_wconcat
_3wpwcscore = []
_3wpwcfile = open(_3wpwcf,'r')
for line in _3wpwcfile:
    if line.startswith('#'):
        pass
    else:
        data= line.split()
        _3wpwctrue.append( bool(float(data[0])) )
        _3wpwcscore.append( float(data[1].strip()) )
_3wpwcfpr, _3wpwctpr, _ = sklearn.metrics.roc_curve(_3wpwctrue,_3wpwcscore)
_3wpwc = sklearn.metrics.roc_auc_score(_3wpwctrue,_3wpwcscore)


fig = plt.figure(figsize=(8,8))
plt.plot(_2npncfpr,_2npnctpr,label='2NPNC (AUC=%.4f)'%(_2npnc),linewidth=8,color='b')
plt.plot(_2npwcfpr,_2npwctpr,label='2NPWC (AUC=%.4f)'%(_2npwc),linewidth=8,color='g')
plt.plot(_2wpncfpr,_2wpnctpr,label='2WPNC (AUC=%.4f)'%(_2wpnc),linewidth=8,color='r')
plt.plot(_2wpwcfpr,_2wpwctpr,label='2WPWC (AUC=%.4f)'%(_2wpwc),linewidth=8,color='k')

plt.plot(_3npncfpr,_3npnctpr,label='3NPNC (AUC=%.4f)'%(_3npnc),linewidth=8,color='b', linestyle='dashed')
plt.plot(_3npwcfpr,_3npwctpr,label='3NPWC (AUC=%.4f)'%(_3npwc),linewidth=8,color='g', linestyle='dashed')
plt.plot(_3wpncfpr,_3wpnctpr,label='3WPNC (AUC=%.4f)'%(_3wpnc),linewidth=8,color='r', linestyle='dashed')
plt.plot(_3wpwcfpr,_3wpwctpr,label='3WPWC (AUC=%.4f)'%(_3wpwc),linewidth=8,color='k', linestyle='dashed')

plt.legend(loc='lower right',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=22)
plt.ylabel('True Positive Rate',fontsize=22)
plt.axes().set_aspect('equal')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('%s_roc.pdf'%outprefix,bbox_inches='tight')
