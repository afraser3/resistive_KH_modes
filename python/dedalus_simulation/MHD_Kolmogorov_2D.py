"""
Dedalus script for 2D strongfield MHD Kolmogorov flow
This script usesa Fourier bases in both the vertical and horozontal directions with fully periodic boundary condtions.

Usage:
    MHD_Kolmogorov_2D.py [options]
    MHD_Kolmogorov_2D.py <config> [options]



Options:
    --Lx_factor=<Lx>                Box size in x direction [default: 2.0]
    --Lz_factor=<Lz>                Box size in z direction [default: 4.0]
    --Nx=<Nx>                       Horizontal resolution [default: 128]
    --Nz=<Nz>                       Vertical resolution [default: 256]

    --HB=<HB>                       Magnetic field strength [default: 0.4]
    --Pm=<MagneticPrandtl>          Magnetic prandtl number [default: 0.1]
    --Re=<Reynolds>                 Reynolds number [default: 20.0]

    --root_dir=<dir>                Root directory for output [default: ./]
    --label=<label>                 Add label to directory name
    --restart=<restart>             File to restart from

"""

import logging
import os
import sys
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
from dedalus.tools.config import config


from logic.output import initialize_output
from logic.checkpointing import Checkpoint
from logic.extras import filter_field
from logic.parsing import  construct_out_dir

logger = logging.getLogger(__name__)

args   = docopt(__doc__)
if args['<config>'] is not None:
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v


resolution_flags=['Nx','Nz']
data_dir = construct_out_dir(args, base_flags=['HB','Pm','Re','Lx_factor','Lz_factor'],label_flags=[], resolution_flags=resolution_flags, parent_dir_flag='root_dir')
logger.info("saving run in: {}".format(data_dir))


# Simulation Parameters
Lx=float(args['--Lx_factor'])*np.pi
Lz=float(args['--Lz_factor'])*np.pi
Nx=int(args['--Nx'])
Nz=int(args['--Nz'])
HB_star=float(args['--HB'])
Pm=float(args['--Pm'])
Re=float(args['--Re'])
Re_m = Re*Pm
aspect=Lx
init_dt = 0.01 * Lx / (Nx)


logger.info("HB = {:2g}, Pm = {:2g}, Re = {:2g} , boxsize={}x{}, resolution = {}x{}".format(HB_star, Pm, Re, Lx, Lz,  Nx, Nz))

# simulation stop conditions
stop_sim_time = np.inf  # stop time in simulation time units
stop_wall_time = np.inf  # stop time in terms of wall clock
stop_iteration = 100000  # stop time in terms of iteration count


# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['phi', 'psi'], time='t')
# Phi is streamfunction, psi is flux function


problem.parameters['Lz'] = Lz
problem.parameters['Lx'] = Lx
problem.parameters['HB_star'] = HB_star
problem.parameters['Reinv'] = 1.0/Re
problem.parameters['Rminv'] = 1.0/Re_m
problem.parameters['MA'] = 1.0/np.sqrt(HB_star)
problem.parameters['aspect']=aspect

problem.substitutions['zeta'] = "dx(dx(phi)) + dz(dz(phi))"
problem.substitutions['J'] = "dx(dx(psi)) + dz(dz(psi))"
problem.substitutions['u'] = "dz(phi)"
problem.substitutions['w'] = "-dx(phi)"
problem.substitutions['Bx'] = "dz(psi)"
problem.substitutions['Bx_tot'] = "1 + Bx"
problem.substitutions['Bz'] = "-dx(psi)"
problem.substitutions['vel_rms']     = 'sqrt(u**2 +  w**2)'
problem.substitutions['b_mag']        = 'sqrt(Bx**2  + Bz**2)'
problem.substitutions['b_mag_tot']        = 'sqrt(Bx_tot**2  + Bz**2)'
problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
problem.substitutions['u_pert'] = "u - plane_avg(u)"
problem.substitutions['phi_pert'] = "phi - plane_avg(phi)"
problem.substitutions['re_stress']="dx(phi_pert)*dz(phi_pert)"
problem.substitutions['max_stress']="dx(psi)*dz(psi)"

#Equations
problem.add_equation("dt(zeta) - Reinv * (dx(dx(zeta)) + dz(dz(zeta))) - HB_star*dx(J)  = -dx(zeta) * dz(phi) + dx(phi) * dz(zeta) + HB_star * (dx(J) * dz(psi) - dx(psi) * dz(J)) + Reinv*cos(z)", condition="(nx!=0) or (nz!=0)")
problem.add_equation("phi = 0", condition="(nx==0) and (nz==0)")
problem.add_equation("dt(psi) - Rminv*J - dx(phi)  = dx(phi)*(dz(psi)) - dx(psi)*dz(phi)")

# Build solver

solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

#Checkpointing 

checkpoint = Checkpoint(data_dir)
checkpoint_min = 30
restart = args['--restart']
if restart is None:
    
    x = domain.grid(0)
    z = domain.grid(1)
    phi = solver.state['phi']
    psi = solver.state['psi']

# Noise initial conditions

    pert = 1e-6
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]
    phi['g'] = pert * noise
    filter_field(phi)
    phi['g'] += -np.cos(z)


    dt = None
    mode = 'overwrite'
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode, iter=5e3)


# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Analysis

analysis_tasks = initialize_output(solver, data_dir, Lx, Lz, plot_boundaries=False, threeD=False, mode="overwrite", slice_output_dt=100, output_dt=50, out_iter=100)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=0.4,
                     max_change=1.5, max_dt=2e-1, threshold=0.1)
CFL.add_velocities(('u', 'w'))
CFL2 = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=0.4,
                     max_change=1.5, min_change=2e-1, max_dt=2e-1, threshold=0.1)
                                                          
CFL2.add_velocities(('Bx/MA', 'Bz/MA'))

#Set up flow tracking for terminal output
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("dx(Bx) + dz(Bz)", name='divB')
flow.add_property("b_mag/MA", name="b_mag")


# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    Re_avg=0
    logger.info('Starting loop')
    start_iter = solver.iteration
    start_run_time = time.time()
    init_time = last_time = solver.sim_time
    while (solver.ok and np.isfinite(Re_avg)):
        dt1 = CFL.compute_dt()
        dt2 = CFL2.compute_dt()

        dt=np.min([dt1, dt2])
        solver.step(dt)
        effective_iter = solver.iteration - start_iter
        effective_sim_time = solver.sim_time - init_time
        if effective_iter % 100 == 0:
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string +=  'Time: {:8.3e}, '.format(effective_sim_time)
            log_string +=  'dt: {:8.3e}, '.format(dt)
            log_string += 'divB: {:8.3e}, '.format(flow.grid_average('divB'))
            log_string += 'dt_ratio: {:8.3e}, '.format(dt/dt2)
            log_string += 'B_rms: {:8.3e}, '.format(flow.volume_average('b_mag'))
            logger.info(log_string)

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

