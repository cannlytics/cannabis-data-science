#-------------------------------------------------------------------------#
#                   Gibbs Sampling with Data Augmentation Algorithm
#                                  Plots
#                             Author: Keegan Skeate
#                       Copyright 2016. All Rights Reserved.
#-------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pylab import *
plt.rc('text', usetex=True) ; plt.rc('font',family='Computer Modern Roman')
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
#-------------------------------------------------------------------------#    
def fig(b):
    b1 = b[:,[0]] ; b2 = b[:,[1]] ; b3 = b[:,[2]] ; b4 = b[:,[3]]  
    weights = np.ones_like(b1)/len(b1)*10
    bins = 100
    x_axis = np.linspace(-0.5,0.5,bins+1)
    max_yticks = 5
    max_xticks = 4
    fig = plt.figure()
    #B1
    ax1 = fig.add_subplot(221)
    plt.xlim(xmin=-0.235, xmax=-0.175)
    density, bins1, patches = ax1.hist(b1,bins=bins,weights=weights, 
                                 alpha=0.4,color='k',
                                 histtype='stepfilled', align='mid')   
    plt.ylabel('Density',fontsize=11)
    plt.annotate(r'$\beta_1$', xy=(-0.1775,0),fontsize=14,ha='left',va='bottom')
    #plt.xlabel(r'$\hat{\beta}_{pt,1}$',fontsize=11)
    ax1.spines["top"].set_visible(False)  
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis='x',which='major',right='off',top='off',bottom='off')
    ax1.tick_params(axis='y',which='major',right='off',top='off',left='off')
    ax1.yaxis.set_major_locator(plt.MaxNLocator(max_yticks))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(max_xticks))
    ax1.xaxis.get_major_ticks()[0].draw = lambda *args:None
    ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    
    #B2
    ax2 = fig.add_subplot(222)
    plt.hist(b2,bins=bins,weights=weights,
                                 alpha=0.4,color='k',
                                 histtype='stepfilled', align='mid')
    plt.annotate(r'$\beta_2$', xy=(0.1325,0),fontsize=14,ha='left',va='bottom')
    plt.xlim(xmax=0.133)
    ax2.spines["top"].set_visible(False)  
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis='x',which='major',right='off',top='off',bottom='off')
    ax2.tick_params(axis='y',which='major',right='off',top='off',left='off')    
    #plt.xlabel(r'$\hat{\beta}_{pt,2}$',fontsize=11)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(max_yticks))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(max_xticks))
    ax2.xaxis.get_major_ticks()[0].draw = lambda *args:None
    ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    
    #B3
    ax3 = fig.add_subplot(223)
    plt.ylabel('Density',fontsize=11)
    plt.hist(b3,bins=bins,weights=weights,
                                 alpha=0.4,color='k',
                                 histtype='stepfilled', align='mid') 
    plt.annotate(r'$\beta_3$', xy=(0.08,0),fontsize=14,ha='left',va='bottom')
    ax3.spines["top"].set_visible(False)  
    ax3.spines["right"].set_visible(False)
    ax3.tick_params(axis='x',which='major',right='off',top='off',bottom='off')
    ax3.tick_params(axis='y',which='major',right='off',top='off',left='off')    
    #plt.xlabel(r'$\hat{\beta}_{pt,2}$',fontsize=11)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(max_yticks))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(max_xticks))
    ax3.xaxis.get_major_ticks()[0].draw = lambda *args:None
    ax3.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    
        # SIGMA
    ax4 = fig.add_subplot(224)
    plt.hist(b4,bins=bins,weights=weights, 
                                 alpha=0.4,color='k',
                                 histtype='stepfilled', align='mid')
    plt.annotate(r'$\sigma^2$', xy=(0.0175,0),fontsize=14,ha='left',va='bottom')
    #plt.xlim(xmin=-0.015, xmax=0.0166)
    ax4.spines["top"].set_visible(False)  
    ax4.spines["right"].set_visible(False)
    ax4.tick_params(axis='x',which='major',right='off',top='off',bottom='off')
    ax4.tick_params(axis='y',which='major',right='off',top='off',left='off')    
    #plt.xlabel(r'$\hat{\beta}_{pt,2}$',fontsize=11)
    ax4.yaxis.set_major_locator(plt.MaxNLocator(max_yticks))
    ax4.xaxis.set_major_locator(plt.MaxNLocator(max_xticks))
    ax4.xaxis.get_major_ticks()[0].draw = lambda *args:None
    ax4.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))


    fig.tight_layout()
    plt.show()
    return fig