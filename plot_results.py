import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import seaborn as sns
import itertools
import csv

base=10

# files to load
save_dir = 'replay-sweep-march-20'
replay_sizes = [1 10 20 50 100 200]
methods_all = ['_outputs/imnet-r_low-lr_5task/dual-prompt/',
                '_outputs/imnet-r_low-lr_5task/atteprompt_len-sweep/',
                '_outputs/imnet-r_low-lr_5task/l2p_len-sweep/_']
labels_all = ['DualPrompt','CODA-P','L2P']
repeat=3

# load sweeep
plen_sweep_all = []
for load_dir in load_dir_all:
    plen_sweep = []
    for c in plen_sizes:
        load_file = load_dir  + str(c) + '/results-acc/global.yaml'
        
        with open(load_file, 'r') as pfile:
            y_ = np.asarray(yaml.safe_load(pfile)['history'])
            try:
                y = y_[:,0:repeat]
            except:
                y = y_
            y = np.mean(y,axis=1)
            acc = np.mean(y,axis=0)
            plen_sweep.append(acc)
    plen_sweep_all.append(plen_sweep)

# save name
outdir = "plots_and_tables/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
outfile = outdir + save_dir

# plot with matplotlib
cmap = plt.get_cmap('jet')
marks = [',', '+', '.', 'o', '*']
max_linspace = len(plen_sweep_all)
colors = cmap(np.linspace(0, 1.0, max_linspace))
plt.figure(figsize=(8,4))
x = np.asarray(plen_sizes)

# plot rehearsal
for i in range(len(plen_sweep_all)):
    plen_sweep = plen_sweep_all[i]
    plt.plot(x, np.asarray(plen_sweep), color = colors[i], linestyle = 'solid', lw=3)
    plt.scatter(x, np.asarray(plen_sweep), color = colors[i], label = labels_all[i], marker = '.',s=500)

# axis and stuff
plt.yticks(np.arange(10, 110, 5),fontsize=14)
plt.xlabel('Prompt Length', fontweight='bold', fontsize=18)
plt.ylabel('$A_n (\%)$', fontweight='bold', fontsize=18)
x_t_l = np.asarray(plen_sizes)
x_t = np.asarray(plen_sizes)
plt.xticks(x_t, [str(plen) for plen in x_t_l],fontsize=14)
plt.ylim(55,80)  
plt.xlim(2,42)
plt.grid()
plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 15})
plt.tight_layout()
plt.savefig(outfile+'.png') 
plt.close()