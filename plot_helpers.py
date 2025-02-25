
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to create seaborn plots that share the same appearance attributes
#   such as: heading, title, x-y labels, and placement of the legend
# 
def render_plot(ax, heading='', subtitle='', xlabel='', ylabel='', rotation=90, plotname='',displaybarlabel=True):
    
    plt.suptitle(heading, fontsize = 12)                        # main heading of the plot
    plt.title(subtitle, fontsize = 8)                          # sub title of the plot
    
    ax.set(xlabel=xlabel, ylabel=ylabel)                        # x and y labels
    
    if displaybarlabel == True:
        for i in ax.containers:
            ax.bar_label(i,)                                    # display values on bars
        
    #ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        
    for patch in ax.patches:                                    # remove bars that do not have any values
        if patch.get_height() == 0: 
            patch.set_visible(False)
    
    if ax.get_legend():                                         # if legend is present, set the location to upper right
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))     
        
    ax.tick_params(rotation=rotation, axis='x')                 # rotate the labels
    
    if len(plotname) > 0:
        filename = 'plots/' + plotname
        ax.get_figure().savefig(filename, bbox_inches='tight')
    

