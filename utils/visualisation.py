
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def per_class_plots(per_class_accuracy, per_class_dist_shift, labels_to_names, class_mapping, epochs_of_interest, steps_of_interest, times_of_interest, replay_size, avg_acc=None, task_acc=None, base_path = 'plots_and_tables/', is_oracle=False):

    colors  = ['r','g', 'b', 'y', 'tomato', 'k', 'c', 'maroon', 'olive', 'm']
    colors = {labels_to_names[class_mapping[k]]:c for k, c in zip(class_mapping.keys(), colors)} | { k: c for k, c in zip(avg_acc.keys(), colors)}

    def plot_utils(data):
        
        to_plot = []
        labels = []
        plot_colors = []

        # print ('hellllloooooooooooooo', data)
        for i, (k, v) in enumerate(data.items()):
            try:
                labels.append(labels_to_names[class_mapping[k]])
            except:
                labels.append(k)
            to_plot.append(v)
            plot_colors.append(colors[labels[-1]])
        
        return to_plot, labels, plot_colors

    if not is_oracle:
        title = 'Replay samples per class: '+ str(replay_size)
    else:
        title = 'Oracle'

    if len(class_mapping)<=10:
        to_plot, labels, plot_colors = plot_utils(per_class_accuracy)     
        plot_multiline(to_plot, labels, plot_colors, xticks = epochs_of_interest, xlabel='epochs', ylabel='accuracy', y_lim_bottom=0.3, title=title, fig_name = base_path+'per_class_accuracy' + '.png', sort_handles=True)
            
        for metric in per_class_dist_shift:    
            to_plot, labels, plot_colors = plot_utils(per_class_dist_shift[metric])
            plot_multiline(to_plot, labels, plot_colors, xticks = epochs_of_interest, xlabel='epochs', ylabel='distribution shift between task one and task two model', y_lim_bottom=0, title=title, fig_name = base_path+'per_class_dist_shift_' + metric + '.png', sort_handles=True)

    if avg_acc is not None:    
        to_plot, labels, plot_colors = plot_utils(avg_acc)     
        plot_multiline(to_plot, labels, plot_colors, xticks = epochs_of_interest, xlabel='epochs', ylabel='average class accuracy', y_lim_bottom=0, title=title, fig_name = base_path+'avg_acc_epoch.png')
        plot_multiline(to_plot, labels, plot_colors, xticks = steps_of_interest, xlabel='steps', ylabel='average class accuracy', y_lim_bottom=0, title=title, fig_name = base_path+'avg_acc_step.png')
        plot_multiline(to_plot, labels, plot_colors, xticks = times_of_interest, xlabel='time (s)', ylabel='average class accuracy', y_lim_bottom=0, title=title, fig_name = base_path+'avg_acc_time.png')

    if task_acc is not None:
        to_plot, labels, plot_colors = plot_utils(task_acc)     
        plot_multiline(to_plot, labels, plot_colors, xticks = epochs_of_interest, xlabel='epochs', ylabel='average task accuracy', y_lim_bottom=0, title=title, fig_name = base_path+'avg_task_accuracy.png')

def plot_multiline(to_plot, labels, plot_colors, xticks, xlabel, ylabel, y_lim_bottom, title, fig_name, sort_handles=False):

    plt.clf()
    max_v = -10000
    # print (to_plot)
    for i, v in enumerate(to_plot):
        label = labels[i]
        plt.plot(range(0, len(v)), v, '.-', label=label, color = plot_colors[i])
        max_v = max(max_v, max(v))
            
    if sort_handles:
        handles,labels = plt.gca().get_legend_handles_labels()
        legend_order = np.array(to_plot)[:, -1].argsort()[::-1]
        handle_order, label_order = np.array(handles)[legend_order], np.array(labels)[legend_order]
        plt.legend(handle_order, label_order)  # To draw legend
    
    x = xticks
    xi = list(range(len(x)))
    plt.xlabel(xlabel)
    plt.xticks(xi, x)

    plt.gca().set_ylim(bottom=y_lim_bottom, top=max_v+0.01)
    
    plt.ylabel(ylabel) 
    plt.title(title)
    
    plt.grid(True)

    plt.savefig(fig_name)

    plt.clf()

def plot_bar(to_plot, plot_colors, xticks, xlabel, ylabel, title, fig_name):

    plt.clf()
    
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=8.5)
    plt.bar(xticks, to_plot, color = plot_colors, edgecolor='k', width = 0.5, align='edge')
    xlocs, xlabs = plt.xticks()

    for i, v in enumerate(to_plot):
        v =  float("{:.2f}".format(v))
        plt.text(xlocs[i] - 0.25, v + 0.01, str(v), size=8.5)

    plt.subplots_adjust(hspace=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)
    
    plt.savefig(fig_name)