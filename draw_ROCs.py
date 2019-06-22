import numpy as np
import pickle
from matplotlib import pyplot as plt


#### Draw all ROC curve together ###
sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
color_list = ['blue', 'red', 'green']

for fidx in range(0,np.size(sr_list)):
    Sub_run = sr_list[fidx]

    para_name = 'ROC_para_' + Sub_run + '.pkl'
    with open(para_name, 'rb') as f:
        fpr, tpr, auc = pickle.load(f)

    fig3 = plt.subplots(ncols=1, figsize=(5,5))
    plt.figure(2)
    label = Sub_run + '_AUC = {:.3f}'.format(auc)

    lc = color_list[int(np.floor(fidx/2))]

    if fidx%2 == 0:
        ls = 'solid'
    else:
        ls = 'dashed'

    plt.plot(fpr, tpr, label=label, linestyle=ls, color=lc)


plt.plot([0, 1], [0, 1], 'k--')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
ROC_n = './ROC_all.png'
plt.savefig(ROC_n)
plt.close()
