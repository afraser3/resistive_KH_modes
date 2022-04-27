# resistive_KH_modes
Code corresponding to Fraser, Cresswell, Garaud 2022 paper on resistive shear flow instabilities in MHD, 
preprint available at [arXiv:2204.10875](https://arxiv.org/abs/2204.10875).

The direcotry python/eigenmode_calculations/ has all the code needed to make figures 1-5. 
Most of it is in kolmogorov_EVP.py, which includes various methods for calculating the eigenmodes of a sinusoidal 
shear flow with a uniform, streamwise magnetic field. 
All the other scripts in that directory use kolmogorov_EVP.py to make various figures 
(including some that aren't in the paper). 

Figures 1, 2, and 4 are made by first running get_and_save_2D_parameter_scans.py to calculate and save various results 
from eigenvalue calculations across different physical parameters to hdf5 files. 
Then, various plotting scripts use those hdf5 files to make figures 1, 2, and 4 
(specifically, the files that start with plot_2D_scans_*). 

Figures 3 and 5 are made using paper_plot_KH_modes.py and paper_plot_resistive_modes.py.

The file python/dedalus_simulation/MHD_Kolmogorov_2D.py is a sample script for using Dedalus 
(specifically d2, i.e., the v2_master branch of Dedalus on GitHub) to run the nonlinear simulation whose results are 
plotted in Figure 10 of the paper. The data used to make Figure 10 is also found here, along with the plotting script paper_plot.py. 
