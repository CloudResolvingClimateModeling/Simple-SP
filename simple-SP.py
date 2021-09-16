#!/usr/bin/python

# Simple SP demonstration.
# A grid of LES models coupled to a global model consisting only of large-scale advection
# Fredrik Jansson, CWI and TU Delft, 2019-2021
#


import numpy 
from omuse.units import units
from omuse.community.dales.interface import Dales
import matplotlib
import matplotlib.pyplot as plt
import os
from amuse.rfi.async_request import AsyncRequestsPool
import subprocess
import pickle
import sys
from scipy.optimize import brentq
import scipy.ndimage.filters
import logging
import time

log = logging.getLogger(__name__)

# Physical constants
pref0 = 1e5 | units.Pa    # Pa reference pressure
rd    = 287.04 | units.J/units.kg/units.K # gas constant for dry air.  J/kg K.
rv    = 461.5 | units.J/units.kg/units.K  # gas constant for water vapor. J/kg K.
cp    = 1004. | units.J/units.kg/units.K  # specific heat at constant pressure (dry air).
rlv   = 2.53e6 | units.J/units.kg   # latent heat for vaporisation
grav  = 9.81  | units.m/units.s**2  # gravity acceleration. m/s^2
mair  = 28.967 | units.g/units.mol  # molar mass of air


# Exner function
def exner(p):
    return (p / pref0) ** (rd / cp)

# inverse Exner function
def iexner(p):
    return (p / pref0) ** (-rd / cp)


qt_forcings = {"sp": Dales.QT_FORCING_GLOBAL,
               "variance": Dales.QT_FORCING_VARIANCE,
               "local": Dales.QT_FORCING_LOCAL,
               "strong": Dales.QT_FORCING_STRONG}

def initBomex(i, j, itot = 25, jtot = 25, dx=100 | units.m, qt_delta = 0 | units.g / units.kg, dirname='dales', nudge=None):
    workdir = '%s_%d_%d'%(dirname,j,i)
    subprocess.call(["rm", workdir, "-rf"]) # remove a previous work-dir if it exists

    inputdir='bomexh' # own bomex case with U velocity in positive direction. bomexh has extra vertical levels.
    d = Dales(inputdir=inputdir, number_of_workers=1, workdir=workdir, channel_type='sockets',
              redirection='none')
              #redirect_stdout_file='dales-output-%s'%workdir,
              #redirect_stderr_file='dales-error-%s'%workdir)
    
    d.parameters_DOMAIN.itot = itot  # number of grid cells in x
    d.parameters_DOMAIN.jtot = jtot  # number of grid cells in y
    d.parameters_DOMAIN.xsize = itot * dx
    d.parameters_DOMAIN.ysize = jtot * dx

    d.parameters_RUN.ladaptive = True

    # Select advection schemes
    d.parameters_DYNAMICS.iadv_mom = 62
    d.parameters_DYNAMICS.iadv_thl = 52
    d.parameters_DYNAMICS.iadv_qt = 52
    d.parameters_DYNAMICS.iadv_tke = 52
    # d.parameters_DYNAMICS.iadv_sv = [52,52]
    d.parameters_RUN.irandom = i*113 # different random seeds for different LES

    # put cross sections planes central, for bubble test
    # note may break if parallel DALES are used
    d.parameters_NAMCROSSSECTION.crossplane = jtot//2
    d.parameters_NAMCROSSSECTION.crossortho = itot//2

    if nudge:
        d.parameters.qt_forcing = qt_forcings[nudge]
        
    d.commit_parameters()
    d.commit_grid()
    
    # optionally apply a bias in qt
    d.fields[:,:,:].QT += i * qt_delta
    return d


# evolve all models in the grid to <time>
def evolve(grid, time):
    pool = AsyncRequestsPool()
    for j in range(len(grid)):
        for i in range(len(grid[j])):
            req = grid[j][i].evolve_model.asynchronous(time, exactEnd=True)
            pool.add_request(req)
    pool.waitall()


# upwind advection of quantity q, with velocity u
# assuming velocity is positive and constant in x, y 
def advect(q, u, dx, dt):
    # periodic boundaries:
    qm = numpy.roll(q.number, 1, axis=1) | q.unit # qm(j,i,k) = q(j,i-1,k), periodic
    dq = (qm - q) * u * dt / dx
    q[:,:,:] += dq 
    # q += dq works with pure numpy but not with quantities - effect does not propagate back to the outside 


def test_advect():
    Q = numpy.zeros((3,4,5)) | units.shu
    U = numpy.zeros((5)) | units.m / units.s
    Q[0,1,2] = 1.0 | units.shu
    U[:] = 10 | units.m / units.s
    print(U)
    print(Q[0,:,2])
    advect(Q, U, 2500 | units.m, 10 | units.s)
    print(Q[0,:,2])
    advect(Q, U, 2500 | units.m, 10 | units.s)
    print(Q[0,:,2])
    advect(Q, U, 2500 | units.m, 10 | units.s)
    print(Q[0,:,2])

# Create a bubble perturbation, given a DALES grid which is used for grid size and coordinates.
# If gaussian=True, a gaussian perturbation is generated, with standard deviation r, otherwise a
# constant perturbation is generated inside a sphere of radius r.
#
# r, center are quantities, i.e. numbers with units.
def make_bubble(grid, r, center, gaussian=False):# r, center are quantities, i.e. numbers with units.
    # array of squared distance to the center
    rr = (grid.x - center[0])**2 + (grid.y - center[1])**2 + (grid.z - center[2])**2
    if gaussian:
        return numpy.exp(-rr/(2*r**2))
    else:
        return numpy.where (rr < r*r, 1, 0)

# Adjust the moisture varibility in an LES domain, so that the specific liquid water 
# profile ql matches the provided ql_ref.
# This cannot be used before the LES has been stepped - otherwise qsat and ql are not defined.
def variability_nudge(les, ql_ref, DT, constantT=False):
    itot, jtot = les.get_itot(), les.get_jtot()
        
    # random field to be used for additive noise when variability is very small
    # want same noise for each horizontal plane, to give correlation between different layers
    R = numpy.random.normal(size=(itot, jtot)) # gaussian random field, mean=0, standard deviation=1
    R -= R.sum()/(itot*jtot)  # adjust average of R to be exactly 0
    
    # possibly add spatial correlation
    # R = scipy.ndimage.filters.gaussian_filter(R, sigma=2, mode='wrap')
    # should normalize so that st-dev = 1 if we want a to be st.dev later on
     
    qsat = les.get_field("Qsat").number
    qt = les.get_field("QT").number
    ql_av = les.get_profile("QL").number
    qt_av = les.get_profile("QT").number
    p = les.get_presf()
#    ql_ref = les.ql_ref.number

    if constantT:
        thl = les.get_field("THL").number
        ql = les.get_field("QL").number
        
    # get ql difference
    # note the implicit k, qt, qt_av, qsat variables
    # returns ql(beta) - ql_ref
    def get_ql_diff(beta):
        result = numpy.maximum((beta * (qt[:, :, k] - qt_av[k]) + qt_av[k] - qsat[:, :, k]), 0).sum() / (itot * jtot) - ql_ref[k]
        return result

    # get ql difference when using additive noise in R
    # note the implicit k, qt, R, qsat variables
    # returns ql(a) - ql_ref
    def get_ql_diff_additive(a):
        result = numpy.maximum((qt[:, :, k] + (a * R[:,:]) - qsat[:, :, k]), 0).sum() / (
                itot * jtot) - ql_ref[k]
        return result
    
    # beta[k] is a factor by which we multiply the qt variability of layer k
    beta_min = 0  # search interval beta_min...beta_max
    beta_max = 5  # above beta_max, switch from multiplicative to additive noise

    beta = numpy.ones(les.parameters_DOMAIN.kmax)
    for k in range(0, les.parameters_DOMAIN.kmax):
        current_ql_diff = get_ql_diff(1)

        if ql_ref[k] > 1e-9:  # significant amount of clouds in the GCM. Nudge towards this amount.
            # print (k, 'significant ql_ref')
            q_min = get_ql_diff(beta_min)
            q_max = get_ql_diff(beta_max)
            if q_min > 0 or q_max < 0:
                log.info("k:%d didn't bracket a zero. qmin:%f, qmax:%f, qt_avg:%f, stdev(qt):%f " %
                         (k, q_min, q_max, numpy.mean(qt[:, :, k]), numpy.std(qt[:, :, k])))
                # seems to happen easily in the sponge layer, where the variability is kept small
                beta[k] = beta_max # take the largest beta, will trigger use of additive noise below.
            else:
                tt = time.time()
                try:
                    beta[k] = brentq(get_ql_diff, beta_min, beta_max)
                except Exception as e:
                    print(e)
                    log.info("k:%d . qmin:%f, qmax:%f, qt_avg:%f, stdev(qt):%f " %
                         (k, q_min, q_max, numpy.mean(qt[:, :, k]), numpy.std(qt[:, :, k])))
                print('brent took %5.2f ms'%((time.time()-tt)*1000))
                
        elif ql_av[k] > ql_ref[k]:  # The GCM says no clouds, or very little, and the LES has more than this.
            # Nudge towards just below saturation.
            i, j = numpy.unravel_index(numpy.argmax(qt[:, :, k] - qsat[:, :, k]), qt[:, :, k].shape)
            beta[k] = (qsat[i, j, k] - qt_av[k]) / (qt[i, j, k] - qt_av[k])
            #log.info(
            #    '%d nudging towards non-saturation. Max at (%d,%d). qt:%f, qsat:%f, qt_av[k]:%f, beta:%f, ql_avg:%f, '
            #    'ql_ref:%f' % (k, i, j, qt[i, j, k].value_in(units.mfu), qsat[i, j, k].value_in(units.mfu),
            #                   qt_av[k].value_in(units.mfu), beta[k], ql[k].value_in(units.mfu), ql_ref[k].value_in(units.mfu)))
            if beta[k] < 0:
                # this happens when qt_av > qsat
                # log.info('  beta<0, setting beta=1 ')
                beta[k] = 1
        else:
            continue  # no clouds, no nudge - don't print anything

        if beta[k] >= beta_max:
            log.info('  beta %f too large at %3d'%(beta[k], k))

            # try additive noise instead
            # add the random noise field defined in the beginning, with amplitude a.
            # solve for the a needed to match ql in this layer 
            a_min = 0
            a_max = 5
            tt = time.time()
            log.info('ql_diff min:%f, max:%f. ql_ref[k] %f  current_ql_diff:%f'%
                     (get_ql_diff_additive(a_min),get_ql_diff_additive(a_max), ql_ref[k], current_ql_diff))
            log.info('ql_av: %f'%(ql_av[k]))
            if ql_ref[k] > ql_av[k]:
                a = brentq(get_ql_diff_additive, a_min, a_max)
                log.info('additive brent took %5.2f ms'%((time.time()-tt)*1000))
                #log.info('  additive noise st.dev a = %f'%a)
                dQT = a * R
                #les.fields[:,:,k].QT += dQT | units.shu              # works
                #les.fields[:,:,k].QT = (qt[:,:,k] + dQT) | units.shu # doesn't work
                qt[:,:,k] += dQT
            else:
                log.info('ql_ref[k] < ql_av[k] in additive nudge, doing nothing. %f %f.'%(ql_ref[k], ql_av[k]) )
            beta[k] = 1 # we don't do any multiplicative nudging on this layer
        else:
            dQT = (beta[k]-1) * (qt[:,:,k] - qt_av[k])
            qt[:,:,k] += dQT
        if constantT:
            # calculate change in theta_l (dTHL) required to keep *temperature*
            # constant, when qt changes
            ql_target = numpy.maximum((qt[:, :, k] - qsat[:, :, k]), 0)
            dQL = ql_target - ql[:,:,k]
            dTHL = - rlv.number / (cp.number * exner(p[k])) * dQL
            thl[:,:,k] += dTHL

    # write qt and thl fields back to the LES
    les.fields.QT = qt
    if constantT:
        les.fields.THL = thl | units.K

    ## gradual adjustment over time   !! NOTE !!  no THL adjustment here yet
    #alpha = (numpy.log(beta) / DT.value_in(units.s)) # .minimum(0.05 | units.s**-1)
    #print ('Setting alpha', alpha)
    #les.set_qt_variability_factor(alpha)

    #qt_std = qt.std(axis=(0, 1))


# Performs a superparameterized simulation, where the large-scale model performs only advection using an upwind scheme.
def run(steps=60, DT=60 | units.s, spinup = 0 | units.s, nx=4, ny=1, n=25, qt_delta=0|units.g/units.kg, name='dales',
        couple=False, bubble=False, bubbleA=1|units.g/units.kg, nudge=None, constantT=False):
    
    dx=100 | units.m     # small-scale grid size
    DX=n*dx              # large-scale grid size = horizontal size of the LES 
                         # note CFL on the large scale: need U * DT < DX. U ~= 10 m/s

    grid = [[initBomex(i,j,dirname=name,itot=n,jtot=n,dx=dx,nudge=nudge) for i in range(nx)] for j in range(ny)]

    print ('Spinup - evolving to ', spinup)
    evolve(grid, spinup)
    
    if bubble:
        b = make_bubble(grid[0][0].fields, r = 500 | units.m, center = (DX/2, DX/2, 800|units.m), gaussian = True)
        grid[0][0].fields[:,:,:].QT += bubbleA * b 
        
    nz = len(grid[0][0].profiles.z)
    
    # large-scale quantities
    QT  = numpy.zeros((ny, nx, nz)) | units.shu
    QL  = numpy.zeros((ny, nx, nz)) | units.shu
    THL = numpy.zeros((ny, nx, nz)) | units.K
    U   = numpy.zeros((nz))         | units.m / units.s

    state = {}
    state['x'] = grid[0][0].fields[:].x.value_in(units.m)
    state['y'] = grid[0][0].fields[:].y.value_in(units.m)
    state['z'] = grid[0][0].fields[:].z.value_in(units.m)
    state['time'] = []
    
    # set large-scale velosity U to U-profile from one of the LES. U is now kept constant.
    U[:]  = grid[0][0].profiles[:].U
    
    for ti in range (0,steps):
        if couple:
            for j in range(ny):
                for i in range(nx):
                    QT[j,i,:]  = grid[j][i].profiles[:].QT # get profiles
                    QL[j,i,:]  = grid[j][i].profiles[:].QL 
                    THL[j,i,:] = grid[j][i].profiles[:].THL            
                    
            print ('QL profile before adv 0')
            print (QL[0,0,:])
            print()
            print ('QL profile before adv 1')
            print (QL[0,1,:])
            
            advect(QT,  U, DX, DT)                         # large-scale advection
            advect(QL,  U, DX, DT)
            advect(THL, U, DX, DT)

            for j in range(ny):                            # set forcings
                for i in range(nx):
                    grid[j][i].forcing_profiles[:].QT  = (QT [j,i,:] - grid[j][i].profiles[:].QT ) / DT
                    grid[j][i].forcing_profiles[:].THL = (THL[j,i,:] - grid[j][i].profiles[:].THL) / DT
                    grid[j][i].set_ref_profile_QL(QL[j,i,:])
                    if nudge == 'variance':
                        variability_nudge(grid[j][i], QL[j,i,:].number, DT, constantT=constantT)
                        
            print ('QL target profile 0')
            print (QL[0,0,:])
            print()
            print ('QL target profile 1')
            print (QL[0,1,:])
            print()
            print ('QL target profile 2')
            print (QL[0,2,:])
            
        print ('Evolving to ', DT*ti + spinup)
        evolve(grid, DT*ti + spinup)

        for j in range(ny):                                # fetch data for saving 
            for i in range(nx):
                for var, unit in (('U',   units.m / units.s),
                                  ('V',   units.m / units.s),
                                  ('QT',  units.shu),
                                  ('QL',  units.shu),
                                  ('THL', units.K),
                                  ('T',   units.K),
                                  ('z',   units.m)):
                    state[(ti,i,j,var)]   = getattr(grid[j][i].profiles, var).value_in(unit)
        state['time'].append((DT*ti).value_in(units.s))
        
    filename='result%s.pickle'%('-coupled' if couple else '')
    with open(filename, 'wb') as f:
        pickle.dump(state, f)    

    for j in range(ny):                                    # stop the models
        for i in range(nx):
            grid[j][i].stop()



def run_single(steps=60, DT=60 | units.s, n=25, nx=4, ny=1, qt_delta=0|units.g/units.kg, name='dales', bubble=False, bubbleA=1|units.g/units.kg):
    dx=100 | units.m     # small-scale grid size
    d = initBomex(0, 0, itot=n*nx, jtot=n*ny, dx=dx, dirname=name)

    DX=n*dx              # large-scale grid size = horizontal size of the small LES tiles
    if bubble:
        b = make_bubble(d.fields, r = 500 | units.m, center = (DX/2, DX/2, 800|units.m), gaussian = True)
        d.fields[:,:,:].QT += bubbleA * b 
    
    for j in range(ny):
        for i in range(nx):
            d.fields[i*n:(i+1)*n, j*n:(j+1)*n, :].QT += i * qt_delta
    #print(d.grid[:,1,2].QT)

    state = {}
    state['x'] = d.fields[:].x.value_in(units.m)
    state['y'] = d.fields[:].y.value_in(units.m)
    state['z'] = d.fields[:].z.value_in(units.m)
    state['time'] = []
    for ti in range (0,steps):
        print ('Evolving to ', DT*ti)
        d.evolve_model(DT*ti, exactEnd=True)
        for var, unit in (('U',   units.m / units.s),
                          ('V',   units.m / units.s),
                          ('QT',  units.shu),
                          ('QL',  units.shu),
                          ('THL', units.K),
                          ('T',   units.K),
                          ('z',   units.m)):
            state[(ti,0,0,var)]   = getattr(d.profiles, var).value_in(unit)
        state['time'].append((DT*ti).value_in(units.s))


    filename='result%s.pickle'%('-single')
    with open(filename, 'wb') as f:
        pickle.dump(state, f)    
    d.stop()
        

# add moist bubble in the leftmost DALES
A = 1.5 | units.g/units.kg

# uncoupled models
#run       (steps=30, DT=60 | units.s, nx=4, ny=1,  couple=False, name='bubble', bubble=True, bubbleA=A)  

# coupled models - regular SP
run       (steps=30, DT=60 | units.s, nx=4, ny=1, couple=True, name='bubble-coupled', bubble=True, bubbleA=A)

# single wide DALES
run_single(steps=30, DT=60 | units.s, nx=4, ny=1, name='bubble-single', bubble=True, bubbleA=A)

# SP with variance nudging at constant thl
# run       (steps=30, DT=60 | units.s, nx=4, ny=1, couple=True, name='bubble-coupled-var', bubble=True, bubbleA=A, nudge='variance')

# SP with variance nudging at constant T
run       (steps=30, DT=60 | units.s, nx=4, ny=1, couple=True, name='bubble-coupled-var-T', bubble=True, bubbleA=A, nudge='variance', constantT=True)





