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

# save name
outdir = "plots_and_tables/" + ins["save_dir"] + '/' + ins["save_name"]
if not os.path.exists(outdir):
    os.makedirs(outdir)
outfile = outdir + '/'

num_x = ins['num_tasks']
y_name=ins["ys"]

for same_plot in [True, False]:
    same_plot_idx = 0
    rows = 2
    cols = 3
    sample_plot_title = outfile+'_main_results'
    if same_plot: plt.figure(figsize=(24,12))

    for metric_prefix in ['acc']:

        acc_flag = metric_prefix == 'huhhh?'

        # Import results
        results_all = {}
        results_all_pt = {}
        results_all_pt_b = {}
        c_index_map = {}
        for name, results_header in ins["results"].items(): 
            c_index_map[results_header["name"]] = int(name)
            try:
                with open(results_header["file"]+'/results-' + metric_prefix + '/global.yaml', 'r') as file:
                    results_all[results_header["name"]] = yaml.safe_load(file)
            
            except:
                print('Could not load '+str(results_header))

        csv_rows = None
        metrics = ['Accuracy']
        for mindex in range(len(metrics)):

            metric = metrics[mindex]
            if metric == 'Accuracy' or acc_flag:

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

                for name, results in results_all.items():
                    

                    y = np.asarray(results['mean'])
                    y_history = np.asarray(results['history'])
                    num_x = ins['num_tasks']
                    
                    if len(y) == 1: 
                        y_long = y * np.ones((num_x,num_x,1)).tolist()
                        y = y * np.ones((num_x,)).tolist()
                        offline.append(True)
                        
                    else:
                        offline.append(False)
                        
                    x_plot.append(x)
                    y_plot.append(y)
                    try:
                        std_plot.append(results['std'])
                    except:
                        std_plot.append(np.std(np.asarray(y_history),axis=1))
                    
                    name_plot.append(name)
                    final_acc.append(y[-1])
                    init_acc.append(y[0])

                print ('x_plot:', x_plot)
                print ('y_plot:',y_plot)
                if len(final_acc) > 1:

                    csv_rows = []
                    row_head = ['Learner']
                    # row_head.extend(['T-' + str(t+1) + ' Acc' for t in range(num_x)])
                    row_head.extend(['Final Acc-Mean'])
                    csv_rows.append(row_head)

                    if acc_flag:
                        for j in range(len(name_plot)):
                            print(name_plot[j])
                            results_row = [name_plot[j]]
                            results_row.extend(['{acc:.1f}'.format(acc=y_plot[j][-1])])
                            csv_rows.append(results_row)

                    if not metric == 'Accuracy' and acc_flag:
                        final_acc = []
                        init_acc = []
                        y_plot_new = []
                        std_plot_new = []
                        for i in range(len(y_plot)):
                            y = y_plot[i]
                            y_metric = []
                            y_std = []
                            y_std.append(0.0)
                            y_plot_new.append(np.asarray(y_metric))
                            std_plot_new.append(np.asarray(y_std))
                            final_acc.append(y_metric[-1])
                            init_acc.append(y_metric[0])
                        y_plot = y_plot_new
                        std_plot = std_plot_new  

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
                        
                    if metric_prefix == 'mem':
                        plt.ylabel(' # Stored Parameters', fontweight='bold', fontsize=18)
                    elif metric_prefix == 'time':
                        plt.ylabel('Time (sec per batch)', fontweight='bold', fontsize=18)
                    elif metric_prefix == 'plastic':
                        plt.ylabel('New Task Accuracy (%)', fontweight='bold', fontsize=18)
                    elif metric_prefix == 'til':
                        plt.ylabel('Task Incremental Accuracy (%)', fontweight='bold', fontsize=18)
                    elif metric_prefix == 'cka':
                        plt.ylabel('CKA', fontweight='bold', fontsize=18)
                    else:
                        plt.ylabel(metric + ' (%)', fontweight='bold', fontsize=18)
                    plt.xlabel('Tasks', fontweight='bold', fontsize=18)
                    # tick_x = ins["split_size"] * np.arange(1+ins['num_tasks'])
                    # if ins['split_size'] > 0:
                    #     tick_x = ins['split_size'] * np.arange(1+ins['num_tasks'])
                    # else:
                    #     tick_x = 100 + ins['split_size'] * np.arange(ins['num_tasks'])
                    #     tick_x = tick_x[::-1]
                    #     tick_x = np.insert(tick_x,0,0)
                    # tick_x_s = []
                    # if len(tick_x) <= 10:
                    #     for tick in tick_x:
                    #         if tick % ins["split_size"] == 0:
                    #             tick_x_s.append(str(tick))
                    #         else:
                    #             tick_x_s.append('')
                    # else:
                    #     for tick in tick_x:
                    #         if tick % (ins["split_size"] * 2) == 0:
                    #             tick_x_s.append(str(tick))
                    #         else:
                    #             tick_x_s.append('')
                    if mindex > 0: plt.ylim(0,10)
                    tick_x = [10,20,30,40,50,60,70,80,90,100]
                    tick_x_s = [1,2,3,4,5,6,7,8,9,10]
                    plt.xticks(tick_x, tick_x_s,fontsize=14)
                    if metric_prefix == 'cka':
                        plt.xlim(2)
                    else:
                        plt.xlim(10)
                    if len(tick_x) <= 3:
                        plt.xlim(95,100)
                    # plt.title(ins["title"], fontweight='bold', fontsize=14)
                    
                    if len(name_plot) > 10:
                        legend_font = 6
                    elif len(name_plot) > 8:
                        legend_font = 7
                    elif len(name_plot) > 6:
                        legend_font = 8
                    else:
                        legend_font = 10
                    if metric_prefix == 'time':
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

        if acc_flag and csv_rows is not None:
            with open(outfile+'metrics.csv', mode='w') as save_file:
                csv_writer = csv.writer(save_file, delimiter=',')
                for t in range(len(csv_rows)): csv_writer.writerow(csv_rows[t])

            # write as latex table rows
            with open(outfile+'LATEX.txt', mode='w') as save_file:
                for item in csv_rows[0]:
                    save_file.write(str(item))
                    save_file.write('   ')
                save_file.write('\n')
                for t in range(1,len(csv_rows)):
                    save_file.write(str(csv_rows[t][0]))
                    save_file.write(' ')
                    for j in range(3):
                        # save_file.write(' & ')
                        # save_file.write('$ ')
                        # save_file.write(csv_rows[t][j*2 + 1])
                        # save_file.write(' \pm ')
                        # save_file.write(csv_rows[t][j*2 + 2])
                        # save_file.write(' $')
                        save_file.write(' & ')
                        if t == 1:
                            best = False
                        else:
                            best = True
                            perf = float(csv_rows[t][j + 1])
                            for t_ in range(2,len(csv_rows)):
                                if j > 0:
                                    if float(csv_rows[t_][j + 1]) < perf:
                                        best = False
                                else:
                                    if float(csv_rows[t_][j + 1]) > perf:
                                        best = False
                        if best:
                            save_file.write('$ \\bm{')
                            save_file.write(csv_rows[t][j + 1])
                            save_file.write('} $')
                        else:
                            save_file.write('$ ')
                            save_file.write(csv_rows[t][j + 1])
                            save_file.write(' $')
                    save_file.write('  \\\\ ')
                    save_file.write('\n')