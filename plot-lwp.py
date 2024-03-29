#!/usr/bin/python

from __future__ import print_function
from __future__ import division

import pickle
import numpy
import matplotlib.pyplot as plt
from matplotlib import ticker
from netCDF4 import Dataset, num2date, date2index # pip install netCDF4


# plot results from simple-SP
# assuming each DALES was run with one process - no merging implemented in this script yet
# Each row is one time point, columns show different DALES.

mplparams = {"figure.figsize" : [3.47, 3.8],    # figure size in inches
                 "figure.dpi"     :  300,     # figure dots per inch
                 "font.size"      :  6,       # this one acutally changes tick labels
                 'svg.fonttype'   : 'none',   # plot text as text - not paths or clones or other nonsense
                 'axes.linewidth' : .5, 
                 'xtick.major.width' : .5,
                 'ytick.major.width' : .5,
                 'font.family' : 'sans-serif',
                 'font.sans-serif' : ['PT Sans'],
    }
plt.rcParams.update(mplparams)

C,R = 4,4 # number of columns and rows in plot

vmin={'lwp' : 0,
      'twp' : 35}
vmax={'lwp' : .2,
      'twp' : 37}

t_start = 60  # s
dtRow   = 120 # s
expname = 'bubble'


def makeplot(dirname, field, t_start, dtRow, vmin, vmax, colorbar=False):
    fig, axes = plt.subplots(R, C, sharex=True, sharey=True, squeeze=False)
    fig.suptitle(dirname + ' ' + field)
    for i in range(C):
        workdir = '%s_%d_%d'%(dirname,0,i)
        filename=workdir+"/"+"cape.x000y000.001.nc"
        print(filename)
        dales_surf = Dataset(filename, "r")
        for j in range(R):
            T = t_start + j * dtRow
            ti = max(T  / (dales_surf["time"][1] -  dales_surf["time"][0]) -1,0)
            # print(T, dales_surf["time"][ti])
            try:
                v=dales_surf[field][ti]
                im=axes[j,i].imshow(v, origin='lower', vmin=vmin,vmax=vmax)
                #print(v.min(), v.max())
            except:
                pass

            # hide all tick marks and labels
            axes[j][i].axis('off') 
            #axes[j][i].xaxis.set_ticks_position('none')
            #axes[j][i].yaxis.set_ticks_position('none')
            
            if i == C-1:
                timestamp = '%3.0f s'%dales_surf["time"][ti]            
                axes[j][i].text(.5,.7,timestamp)
            
        # calculate average over time
        for ti in range(0, 10):
            lwp_avg[(dirname, i, ti, field)] = dales_surf[field][ti][:,:].sum()

    if colorbar: # add color bars for this plot
        cax = fig.add_axes([0.2, 0.05, 0.3, 0.03]) #horizontal colorbar
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        tick_locator = ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label('kg/m$^2$')
    
    plt.subplots_adjust(left=.01, top=.99, bottom=.1, right=.99, wspace=.01, hspace=.01)
    plt.savefig(dirname+'_'+field+'.png')
    plt.savefig(dirname+'_'+field+'.svg')

lwp_avg = {}   # lwp_avg[('name',column,time_index)] = sum(lwp) / (itot*jtot)

E = ('-coupled', '-coupled-var-T')  # '-coupled', '-coupled-var', '-coupled-var-T', '-coupled-strong':
for field in 'lwp', 'twp':
    for extension in E:
        dirname = expname+extension
        colorbar = (extension == '-coupled-var-T')
        makeplot(dirname, field, t_start, dtRow, vmin[field], vmax[field], colorbar)


# Plotting the single wide LES run
    
#C=2  # show also qt cross-section
C=1
dirname=expname+'-single'

for field in 'lwp', 'twp':
    fig, axes = plt.subplots(R, C, sharex=True, sharey=False, squeeze=False)
    fig.suptitle(dirname + ' ' + field)
    workdir = '%s_%d_%d'%(dirname,0,0)
    filename=workdir+"/"+"cape.x000y000.001.nc"
    crossxz_filename=workdir+"/"+"crossxz.x000y000.001.nc"
    # print(filename)
    dales_surf = Dataset(filename, "r")
    dales_crossxz = Dataset(crossxz_filename, "r")
    for i in range(C):
        if i == 0: # plot lwp from above
            for j in range(R):
                T = t_start + j * dtRow
                ti = max(T / (dales_surf["time"][1] -  dales_surf["time"][0]) -1,0)
                #print(T, dales_surf["time"][ti])
                timestamp = '%3.0f s'%dales_surf["time"][ti] 
                lwp=dales_surf[field][ti]
                im=axes[j,i].imshow(lwp, origin='lower', vmin=vmin[field],vmax=vmax[field])
                axes[j,i].text(.5,.7,timestamp)
                axes[j,i].axis('off')
        if i == 1: # plot qt cross section from side
            for j in range(R):
                T = t_start + j * dtRow
                ti = max(T  / (dales_surf["time"][1] -  dales_surf["time"][0]) -1,0)
                #print(T, dales_surf["time"][ti])
                lwp=dales_crossxz["qtxz"][ti]
                im_v=axes[j,i].imshow(lwp, origin='lower', vmin=0,vmax=0.02)

    cax = fig.add_axes([0.2, 0.05, 0.3, 0.03]) #horizontal colorbar
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator

    cbar.update_ticks()
    cbar.set_label('kg/m$^2$')
    plt.subplots_adjust(left=.01, top=.99, bottom=.1, right=.99, wspace=.01, hspace=.01)
    
    plt.savefig(dirname+'_'+field+'.png')
    plt.savefig(dirname+'_'+field+'.svg')

if False:
    plt.show()        


