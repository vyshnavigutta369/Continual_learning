import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import seaborn as sns
import itertools
import csv
from sklearn.metrics import roc_auc_score

MIN_LINSPACE = 8
# seaborn defaults
sns.set()

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Consolas"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

# Arg Parser
parser = argparse.ArgumentParser()
parser.add_argument('--ins', type=str, default="plot_ins/ins.yaml",
                         help="Instructions for plotting")
parser.add_argument('--std', default=False, action='store_true', help='Plot standard deviation')                        
args = parser.parse_args()

# Import plot settings
with open(args.ins, 'r') as file:
    ins = yaml.safe_load(file)

# print ('hello: ', ins)
# save name
outdir = "plots_and_tables/" + ins["save_dir"] + '/' + ins["save_name"]
if not os.path.exists(outdir):
    os.makedirs(outdir)
outfile = outdir + '/'

# num_x = ins['num_tasks']
# y_name=ins["ys"]

for same_plot in [True, False]:
    same_plot_idx = 0
    rows = 2
    cols = 3
    sample_plot_title = outfile+'_main_results'
    if same_plot: plt.figure(figsize=(24,12))

    for metric_prefix in ['acc']:

        acc_flag = metric_prefix == 'acc'

        # Import results
        results_all = {}
        results_all_pt = {}
        results_all_pt_b = {}
        c_index_map = {}
        for name, results_header in ins["results"].items(): 
            c_index_map[results_header["name"]] = int(name)
            print (results_header["file"])
            
            # with open(results_header["file"]+'/results-' + metric_prefix + '/global.yaml', 'r') as file:
            with open(results_header["file"], 'r') as file:
                print ('helllo: ', results_header["file"])
                results_all[results_header["name"]] = yaml.safe_load(file)
            
            # except:
            #     print('Could not load '+str(results_header))

    csv_rows = None
    metrics = ['acc', 'time']
    for mindex in range(len(metrics)):

        metric_prefix =  metrics[mindex]

        metric = metrics[mindex]
        print (metric)
        if metric == 'acc' or acc_flag:

            # get x, y, and legend names from results
            # wait to plot until parsed all results - want
            # to plot in order of decreasing final accuracy for easy
            # interpretation
            #
            
            y_plot = []
            y_plot_pt = []
            y_plot_pt_b = []
            std_plot = []
            name_plot = []
            final_acc = []
            init_acc = []
            offline = []
            x_plot=[]
            y_plot=[]

            for name, results in results_all.items():
                
                
                y = np.asarray(results[metric])
                x = np.asarray(results['steps'])
                offline.append(False)
                
                # y = np.asarray(list(y).insert(0,0))
                ytemp = list(y)
                ytemp.insert(0,0)
                y = np.asarray(ytemp)
                # x=  np.asarray(results['steps'])


                x_plot.append(x)
                y_plot.append(y)
                
                name_plot.append(name)
                final_acc.append(y[-1])
                init_acc.append(y[0])

            # print ('final_acc:', final_acc)
            # print ('y_plot:',y_plot)
            # print ('x_plot:',x_plot)
            if len(final_acc) > 1:

                # array for sorting
                final_acc = np.asarray(final_acc)
                init_acc = np.asarray(init_acc)
                # plot with matplotlib
                cmap = plt.get_cmap('jet')
                marks = ['p', 'D', 'X', 'o', '*','v']
                lines = ['solid','dashed']
                max_linspace = max(MIN_LINSPACE, len(c_index_map.items()))
                colors = cmap(np.linspace(0, 1.0, max_linspace))
                if same_plot:
                    same_plot_idx += 1
                    if same_plot_idx > 6:
                        same_plot_idx = 1
                        rows = 1
                        sample_plot_title = outfile+'_side_results'
                        plt.figure(figsize=(24,6))
                    plt.subplot(rows, cols, same_plot_idx)
                else:
                    plt.figure(figsize=(8,4))
                for i in range(len(name_plot)):

                    # get next highest accuracy
                    j = np.argsort(final_acc)[-i-1]
                    c_index = max_linspace - c_index_map[name_plot[j]]-1

                    # if offline
                    if offline[j]:

                        # add to plot
                        plt.plot(x_plot[j], y_plot[j], color = colors[c_index], label = name_plot[j], linestyle = lines[c_index % len(lines)], lw=2)
                        
                        # add standard deviation
                        if args.std:
                            sigma = np.asarray(std_plot[j])
                            plt.fill_between(x_plot[j], y_plot[j]+sigma, y_plot[j]-sigma, facecolor=colors[c_index], alpha=0.3)

                    # else
                    else:

                        # add to plot
                        plt.plot(x_plot[j], y_plot[j], color = colors[c_index], lw=2, linestyle = lines[c_index % len(lines)])
                        plt.scatter(x_plot[j], y_plot[j], color = colors[c_index], label = name_plot[j], marker = marks[c_index % len(marks)],s=50)
                        
                        # add standard deviation
                        if args.std:
                            sigma = np.asarray(std_plot[j])
                            plt.fill_between(x_plot[j], y_plot[j]+sigma, y_plot[j]-sigma, facecolor=colors[c_index], alpha=0.3)

                            
                # axis and stuff
                if acc_flag:
                    plt.yticks(np.arange(-10, 110, 10),fontsize=14)
                    miny=min(60,np.floor(np.amin(np.asarray(y_plot))/10)*10)
                    maxy=max(101,np.ceil(np.amax(np.asarray(y_plot))/10)*10)
                    plt.ylim(int(miny),int(maxy))
                    # plt.ylim(np.floor(np.amin(np.asarray(y_plot))/10)*10, np.ceil(np.amax(np.asarray(y_plot))/10)*10)
                    
                #
                if metric_prefix == 'acc':
                    plt.ylabel(' # Avg class accuracy', fontweight='bold', fontsize=18)
                elif metric_prefix == 'time':
                    plt.ylabel('Time (sec)', fontweight='bold', fontsize=18)
                elif metric_prefix == 'steps':
                    plt.ylabel('Steps', fontweight='bold', fontsize=18)
                else:
                    plt.ylabel(metric + ' (%)', fontweight='bold', fontsize=18)
                plt.xlabel('Steps', fontweight='bold', fontsize=18)
                
                if metric == 'time':
                    # print (y_plot)
                    if mindex > 0: plt.ylim(0, np.max(np.array(y_plot))+0.5)
                    plt.yticks(np.arange(0, 5000, 500),fontsize=14)
                else:
                    if mindex > 0: plt.ylim(0,10)
                # tick_x = [10,20,30,40,50,60,70,80,90,100]
                
                tick_x = list(x_plot[0])
                # tick_x_s = [1,2,3,4,5,6,7,8,9,10,11,12]
                tick_x_s = list(x_plot[0])
                plt.xticks(tick_x, tick_x_s,fontsize=14)
                if metric_prefix == 'cka':
                    plt.xlim(2)
                else:
                    plt.xlim(12)
                if len(tick_x) <= 3:
                    plt.xlim(95,100)
                plt.ylim(0, np.max(np.array(y_plot))+0.5)
                
                
                if len(name_plot) > 10:
                    legend_font = 6
                elif len(name_plot) > 8:
                    legend_font = 7
                elif len(name_plot) > 6:
                    legend_font = 8
                else:
                    legend_font = 10
                if metric_prefix == 'acc':
                    # plt.legend(loc='lower right', prop={'weight': 'bold', 'size': legend_font})
                    plt.legend(loc='upper left', prop={'weight': 'bold', 'size': legend_font})
                elif metric_prefix == 'time':
                    plt.legend(loc='upper left', prop={'weight': 'bold', 'size': legend_font})
                elif metric == 'Task FGT' or metric == 'Class FGT':
                    plt.legend(loc='upper left', prop={'weight': 'bold', 'size': legend_font})
                elif not acc_flag:
                    plt.legend(loc='lower right', prop={'weight': 'bold', 'size': legend_font})
                else:
                    plt.legend(loc='lower left', prop={'weight': 'bold', 'size': legend_font})
                plt.grid()
                
                if same_plot:
                    plt.grid()
                    plt.savefig(sample_plot_title+'.png') 
                    print(outfile)  
                else:
                    plt.tight_layout()
                    if acc_flag:
                        plt.grid()
                        plt.savefig(outfile+metric+'.png') 
                        print(outfile)    
                        plt.close()
                    else:
                        plt.grid()
                        plt.savefig(outfile+metric_prefix+'.png') 
                        print(outfile)    
                        plt.close()

        # if acc_flag and csv_rows is not None:
        #     with open(outfile+'metrics.csv', mode='w') as save_file:
        #         csv_writer = csv.writer(save_file, delimiter=',')
        #         for t in range(len(csv_rows)): csv_writer.writerow(csv_rows[t])

        #     # write as latex table rows
        #     with open(outfile+'LATEX.txt', mode='w') as save_file:
        #         for item in csv_rows[0]:
        #             save_file.write(str(item))
        #             save_file.write('   ')
        #         save_file.write('\n')
        #         for t in range(1,len(csv_rows)):
        #             save_file.write(str(csv_rows[t][0]))
        #             save_file.write(' ')
        #             for j in range(3):
        #                 # save_file.write(' & ')
        #                 # save_file.write('$ ')
        #                 # save_file.write(csv_rows[t][j*2 + 1])
        #                 # save_file.write(' \pm ')
        #                 # save_file.write(csv_rows[t][j*2 + 2])
        #                 # save_file.write(' $')
        #                 save_file.write(' & ')
        #                 if t == 1:
        #                     best = False
        #                 else:
        #                     best = True
        #                     perf = float(csv_rows[t][j + 1])
        #                     for t_ in range(2,len(csv_rows)):
        #                         if j > 0:
        #                             if float(csv_rows[t_][j + 1]) < perf:
        #                                 best = False
        #                         else:
        #                             if float(csv_rows[t_][j + 1]) > perf:
        #                                 best = False
        #                 if best:
        #                     save_file.write('$ \\bm{')
        #                     save_file.write(csv_rows[t][j + 1])
        #                     save_file.write('} $')
        #                 else:
        #                     save_file.write('$ ')
        #                     save_file.write(csv_rows[t][j + 1])
        #                     save_file.write(' $')
        #             save_file.write('  \\\\ ')
        #             save_file.write('\n')