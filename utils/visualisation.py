
import matplotlib.pyplot as plt
import numpy as np

def plot_multiline(to_plot, labels, plot_colors, xticks, xlabel, ylabel, y_lim_bottom, title, fig_name, sort_handles=False):

    plt.clf()
    max_v = -10000
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
    plt.savefig(fig_name)