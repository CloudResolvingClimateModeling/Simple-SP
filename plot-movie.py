#!/usr/bin/python

from __future__ import print_function
from __future__ import division

import pickle
import numpy
import matplotlib.pyplot as plt
from matplotlib import ticker
from netCDF4 import Dataset, num2date, date2index # pip install netCDF4
import sys

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# plot results from simple-SP
# assuming each Dales was run with one process - no merging implemented in this script yet

# Uses moviepy to create the movie directly, without saving individual frames as image files
# Note: there is a memory leak, probably either the netcdf objects or the matplotlib figures
# stay in memory.

mplparams = {"figure.figsize" : [8.5, 3.6],    # figure size in inches
                 "figure.dpi"     :  300,     # figure dots per inch
                 "font.size"      :  9,       # this one acutally changes tick labels
                 'svg.fonttype'   : 'none',   # plot text as text - not paths or clones or other nonsense
                 'axes.linewidth' : .5, 
                 'xtick.major.width' : .5,
                 'ytick.major.width' : .5,
                 'font.family' : 'sans-serif',
                 'font.sans-serif' : ['PT Sans'],
    }
plt.rcParams.update(mplparams)

# value ranges for the two fields
vmin={'lwp' : 0,
      'twp' : 35}
vmax={'lwp' : .2,
      'twp' : 37}

expname = 'bubble'

# plot a single LES in an existing axis
def plot_field (ax, workdir, ti, field='lwp', bar=False):
    filename=workdir+"/"+"cape.x000y000.001.nc"
    dales_surf = Dataset(filename, "r")
    v=dales_surf[field][ti]
    im=ax.imshow(v, origin='lower', vmin=vmin[field],vmax=vmax[field])
    if bar:
        ax.axvline(x=-.5)
    return im, dales_surf["time"][ti]

def add_color_bar(fig, pos, im):
    cax = fig.add_axes(pos) #horizontal colorbar
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    #cbar.set_label('kg/m$^2$')

def plot(ti):
    fig = plt.figure()

    Sx = 1/9  # figures are square, but the coordinate system maps  0...1 to the whole vertical or horizontal range
    Sy = 1/3  # which is not square

    m=.06  # left margin
    b = .15 # bottom margin
    R = (1-b) / 3 # row height

    # create axes
    # LWP, left
    a1 =  plt.axes((m+   0,    b+2*R, 4*Sx, Sy))                       # top     single LES
    a2 = [plt.axes((m+i*Sx,    b+R,     Sx, Sy)) for i in range(4)]   # middle   4 frames
    a3 = [plt.axes((m+i*Sx,    b,       Sx, Sy)) for i in range(4)]   # bottom   4 frames   

    #TWP, right
    b1 =  plt.axes((m/2+.5,        b+2*R, 4*Sx, Sy))                      # top     single LES 
    b2 = [plt.axes((m/2+.5+i*Sx,   b+R,     Sx, Sy)) for i in range(4)]  # middle   4 frames                 
    b3 = [plt.axes((m/2+.5+i*Sx,   b,       Sx, Sy)) for i in range(4)]  # bottom   4 frames   
    # plt.axes((left, bottom, width, height))


    im_lwp,time = plot_field(a1, expname+f'-single_0_0', ti, 'lwp')
    im_twp,time = plot_field(b1, expname+f'-single_0_0', ti, 'twp')
    for i in range(4):
        plot_field(a2[i], expname+f'-coupled_0_{i}', ti, 'lwp', bar=(i!=0))
        plot_field(a3[i], expname+f'-coupled-var-T_0_{i}', ti, 'lwp', bar=(i!=0))
        plot_field(b2[i], expname+f'-coupled_0_{i}', ti, 'twp', bar=(i!=0))
        plot_field(b3[i], expname+f'-coupled-var-T_0_{i}', ti, 'twp', bar=(i!=0))
    add_color_bar (fig, [m  +0.03, 0.07, 0.25, 0.04], im_lwp)
    add_color_bar (fig, [m/2+0.53, 0.07, 0.25, 0.04], im_twp)
    plt.text(m+0.31, 0.04, 'Liquid water path\nkg/m$^2$', transform=fig.transFigure)
    plt.text(m/2+0.81, 0.04, 'Total water path\nkg/m$^2$', transform=fig.transFigure)
    plt.text(m/2+0.89, 0.14, f'{int(time/60)} min', transform=fig.transFigure)

    o=.15
    plt.text(0.005, b+3*R-o, 'Single\nmodel', transform=fig.transFigure)
    plt.text(0.005, b+2*R-o, 'SP', transform=fig.transFigure)
    plt.text(0.005, b+1*R-o, 'SP with\nnudging', transform=fig.transFigure)
    
    for ax in [a1, *a2, *a3, b1, *b2, *b3]:
        ax.axis('off') 

#    plt.show()
#    sys.exit()
    img = mplfig_to_npimage(fig)
    fig.clear()
    del(fig)
    return img 

# find number of frames
dales_surf = Dataset(expname+f'-single_0_0'+"/"+"cape.x000y000.001.nc", "r")
frames = len(dales_surf["time"][:])

fps = 20
dur = frames/fps

print(f'{frames} frames, {fps} fps, duration {dur} s.')

def make_frame(t):
    ti = int(fps*t)
    return plot(ti)


animation = VideoClip(make_frame, duration=dur)
#animation.write_gif('animation.gif', fps=fps)
animation.write_videofile('simple-sp.mp4', fps=fps)  # preset='medium'


