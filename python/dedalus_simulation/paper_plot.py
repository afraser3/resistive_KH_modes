import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import dedalus as de
from pylab import *
import matplotlib.patches as patches
rcParams['figure.dpi'] = 200.
rcParams['text.usetex']=True
rcParams.update({'font.size': 12})
import kolmogorov_EVP
from palettable.colorbrewer.qualitative import Dark2_3
from palettable.colorbrewer.qualitative import Dark2_6_r
from palettable.colorbrewer.diverging import PiYG_11
from palettable.colorbrewer.diverging import PuOr_11_r

def scalar_time_avg(scalar_array, time_array):
    length=3*len(time_array)/4
    avg=np.mean(scalar_array[-int(length):])
    return avg

def fft_scalar(array, Lz_factor):
    index=int(Lz_factor/2)
    return 2*(np.fft.rfft(array)[index])/len(array)
    
def get_growth_rate(Hb, re, pm, Lx):
    ks=np.arange(0.00001, 0.2, 0.0001)
    delta=0
    M2=Hb
    Re=re
    Pm=pm
    Rm=Re*Pm
    N=7
    gamma_value=[]
    for i in range(len(ks)):
        k_value=ks[i]
        gamma=kolmogorov_EVP.gamfromparams(delta, M2, Re, Rm, k_value, N, ideal=False, withmode=False)
        gamma_value.append(gamma)
    gamma_array=np.array(gamma_value)
    kbox=4*pi/Lx
    gamma_true=kolmogorov_EVP.gamfromparams(delta, M2, Re, Rm, kbox, N, ideal=False, withmode=False)
    return gamma_array, gamma_true
    

def find_true_values(profile_file, input_re, input_pm, input_hb):
    profiles_file=profile_file
    profiles=h5py.File(profiles_file, 'r')
    u=np.array(profiles['u'])
    time=np.array(profiles['u_times']).flatten()[::2]
    time2=time[-int(len(time)/2):]
    avg_u=np.trapz(u[-int(len(time)/2):],x=time[-int(len(time)/2):],axis=0)/(time2[-1]-time2[0])
    utrue=-np.imag(fft_scalar(avg_u,4))
    z=np.array(profiles['z'])
    Re=input_re
    Pm=input_pm
    Rm=Re*Pm
    Hb=input_hb
    hb_true=Hb/(utrue**2)
    re_true=Re*(utrue)
    rm_true=re_true*Pm
    return utrue, re_true, hb_true, rm_true


def time_and_planar_avg_u(profile_file1, fig_name=None, profile_file2=None, profile_file3=None):
    profiles_file=profile_file1
    profiles=h5py.File(profiles_file, 'r')
    u=np.array(profiles['u'])
    z=np.array(profiles['z'])
    time=np.array(profiles['u_times']).flatten()[::2]
    time2=time[-int(len(time)/2):]
    avg_u=np.trapz(u[-int(len(time)/2):],x=time[-int(len(time)/2):],axis=0)/(time2[-1]-time2[0])
    
    if fig_name is not None:
        plt.figure()
        plt.plot(z, avg_u)
        plt.xlabel('$z$')
        plt.ylabel('$U$')

        if profile_file2 is not None:
            profiles_file2=profile_file2
            profiles2=h5py.File(profiles_file2, 'r')
            u2=np.array(profiles2['u'])
            z2=np.array(profiles2['z'])
            time_2=np.array(profiles2['u_times']).flatten()[::2]
            time2_2=time_2[-int(len(time_2)/2):]
            avg_u2=np.trapz(u2[-int(len(time_2)/2):],x=time_2[-int(len(time_2)/2):],axis=0)/(time2_2[-1]-time2_2[0])
            plt.plot(z2, avg_u2)
        if profile_file3 is not None:
            profiles_file3=profile_file3
            profiles3=h5py.File(profiles_file3, 'r')
            u3=np.array(profiles3['u'])
            z3=np.array(profiles3['z'])
            time_3=np.array(profiles3['u_times']).flatten()[::2]
            time2_3=time_3[-int(len(time_2)/2):]
            avg_u3=np.trapz(u3[-int(len(time_2)/2):],x=time_3[-int(len(time_2)/2):],axis=0)/(time2_3[-1]-time2_3[0])
            plt.plot(z3, avg_u3)
        plt.savefig(str(fig_name))
        plt.show()
    return avg_u, z

def energy_from_linear_modes(re, hb, pm, lx, use_box=True):

    Pm = pm
    HB = hb
    Re = re
    Lx = lx
    Rm = Pm * Re
    delta = 0.0
    N = 33
    ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))
    ks = np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
                   np.linspace(0.05, 0.2, num=50, endpoint=False))

    gammas = kolmogorov_EVP.gamma_over_k(delta, HB, Re, Rm, ks, N)
    maxind = np.argmax(gammas)
    if use_box:
        kmax = 2*np.pi/Lx
    else:
        kmax = ks[maxind]
    gammax = gammas[maxind]
    lmat = kolmogorov_EVP.Lmat(delta, HB, Re, Rm, kmax, N)
    w, v = np.linalg.eig(lmat)
    ind1 = np.argmax(-np.imag(w))  # index in w,v of fastest-growing mode
    omega1 = w[ind1]
    full_mode1 = v[:, ind1]
    phi1 = full_mode1[::2]
    psi1 = full_mode1[1::2]
    norm1_phi = phi1[int(len(ns)/2)+1]
    norm1_psi = psi1[int(len(ns)/2)+1]
    phi1 = phi1/norm1_phi
    psi1 = psi1/norm1_phi
    TE1 = kolmogorov_EVP.energy_from_streamfunc(phi1, kmax, ns) + HB*kolmogorov_EVP.energy_from_streamfunc(psi1, kmax, ns)
    phi1 = phi1/np.sqrt(TE1)
    psi1 = psi1/np.sqrt(TE1)
    KE1 = kolmogorov_EVP.energy_from_streamfunc(phi1, kmax, ns)
    ME1 = HB*kolmogorov_EVP.energy_from_streamfunc(psi1, kmax, ns)
    return KE1, ME1
    
def dynamics_plot(fig_name, slice_file, trace_file, profile_file, dynamics_panel1, dynamics_panel2, dyn1_label, dyn2_label, Lx, Lz, Nx, Nz, input_re, input_hb, input_pm,  energy_ylim, disp_ylim, disp_xlim, inset_ylim, inset_xlim, specs_inset, add_energy_modes=True):
    rcParams.update({'font.size': 13})
    plot_ind=13
    slices_file=slice_file
    traces_file=trace_file
    profiles_file=profile_file
    Re = input_re
    Hb = input_hb
    Pm = input_pm
    ut, ret, hbt, rmt=find_true_values(profile_file, Re, Pm, Hb)
    print('u true =', ut)
    print('re true =', ret)
    print('hb true =', hbt)
    task1=dynamics_panel1
    task2=dynamics_panel2
    label1=dyn1_label
    label2=dyn2_label
    lz = Lz
    lx = Lx
    kbox=(4*np.pi)/Lx
    nx = Nx
    nz = Nz
    ks=np.arange(0.00001, 0.2, 0.0001)
    disp_rel, gamma_plot=get_growth_rate(Hb,Re,Pm,Lx)
    true_disp_rel=get_growth_rate(hbt, ret, Pm,Lx)[0]
    KE_modes, ME_modes=energy_from_linear_modes(ret, hbt, Pm, lx, use_box=False)
    u_profile, z_profile =time_and_planar_avg_u(profiles_file)
    fake_z=np.arange(0,Lz,0.0001)
    fake_u_profile=ut*np.sin(z_profile)
    with h5py.File(slices_file, 'r') as f:
        print(f.keys())
        dyn1 = f['tasks'][str(task1)][plot_ind,:].squeeze()
        dyn2 = f['tasks'][str(task2)][plot_ind,:].squeeze()
        time= f['scales']['sim_time'][plot_ind].squeeze()
    print(time)
    fig = plt.figure(figsize=(5, 3))
    inset_specs=specs_inset
    cax_specs = [(0.65,1.29,0.25,0.06),(0.9, 1.29,0.25,0.06)]
    ax_specs = [( 0.65,-0.56,0.25,1.85),(0.9, -0.56,0.25,1.85),(0,0.85,0.5,0.5),(0,-0.55,0.5,0.5), (0,0.15,0.5,0.5)]
    axs  = [fig.add_axes(spec) for spec in ax_specs]
    caxs = [fig.add_axes(spec) for spec in cax_specs]
        # Dedalus domain for interpolation
    x_basis = de.Fourier('x',   nx, interval=[0, lx], dealias=1)
    z_basis = de.Fourier('z', nz, interval=[0, lz], dealias=1)
    vert_domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
    vert_transverse=de.Domain([z_basis, x_basis], grid_dtype=np.float64)
    hires_scales = 20

    vert_field  = vert_domain.new_field()
    trans_field = vert_transverse.new_field()

        
    x = vert_domain.grid(0, scales=hires_scales)
    x2 = vert_transverse.grid(0, scales=hires_scales)
       

    zz_tb, xx_tb = np.meshgrid(vert_domain.grid(-1, scales=hires_scales), x)
    xx_tb, zz_tb = np.meshgrid(vert_transverse.grid(-1, scales=hires_scales), x2)
    x_fake=np.linspace(0,lx,256)
    z_fake=np.linspace(0,lz,64)
    xx_fake, zz_fake = np.meshgrid(x_fake, z_fake)
    vert_field['g'] = dyn1
    trans_field['g'] = vert_field['g'].T
    trans_field.set_scales(hires_scales, keep_data=True)
    pT = axs[0].pcolormesh(zz_tb,  xx_tb,  np.copy(trans_field['g']), label='$u_{z}$', cmap=PiYG_11.mpl_colormap, rasterized=True, shading="nearest")
 
    
    trans_field.set_scales(1)
    axs[1].set_yticks([])
    axs[0].set_ylabel('$z$')
    axs[0].set_xlabel('$x$')
    axs[1].set_xlabel('$x$')
    axs[0].yaxis.set_label_position("left")

    vert_field['g'] = dyn2
    trans_field['g'] = vert_field['g'].T
    trans_field.set_scales(hires_scales, keep_data=True)

    pB = axs[1].pcolormesh(zz_tb,  xx_tb,  np.copy(trans_field['g']),  cmap=PuOr_11_r.mpl_colormap ,rasterized=True, shading="nearest")

    trans_field.set_scales(1)

    
    traces=h5py.File(traces_file, 'r')

    KE_background=np.array(traces['KE_x']) - np.array(traces['KE_x_pert'])
    ME_background=np.array(traces['ME_x']) - np.array(traces['ME_x_pert'])

    KE_tot=np.array(traces['KE_x_pert']) + np.array(traces['KE_z'])
    ME_tot=np.array(traces['ME_x_pert']) + np.array(traces['ME_z'])




    cbar_B = plt.colorbar(pB, cax=caxs[1], orientation='horizontal', ticks=[-0.15,0,0.15])
    cbar_T = plt.colorbar(pT, cax=caxs[0], orientation='horizontal',ticks=[-0.2,0,0.2])
    cbar_T.set_ticklabels([r"$-0.2$", r"$0$", r"$0.2$"])
    cbar_B.set_ticklabels([r"$-0.15$", r"$0$", r"$0.15$"])


    energy_ylim_max=energy_ylim
    disp_ylim_max=disp_ylim
    disp_xlim_max=disp_xlim
    inset_ylim_max=inset_ylim
    inset_xlim_max=inset_xlim
    scalar_time=np.array(traces['sim_time'])
    KE_growth=1e-15*np.exp(2*gamma_plot*scalar_time)

    three4_time=scalar_time[-int(3*len(scalar_time)/4)]
    print(three4_time)
    print(scalar_time[-1])
    caxs[1].text(0.5, 0.6, label2, transform=caxs[1].transAxes, va='center', ha='center', fontsize=14)
    caxs[0].text(0.5, 0.6, label1, transform=caxs[0].transAxes, va='center', ha='center', fontsize=14)
    caxs[0].xaxis.set_ticks_position('top')
    caxs[0].xaxis.set_label_position('top')
    caxs[1].xaxis.set_ticks_position('top')
    caxs[1].xaxis.set_label_position('top')
    
    pGKE = axs[2].plot(np.array(traces['sim_time']), KE_tot, label='$\mathrm{KE}_\mathrm{pert}$', color=Dark2_6_r.mpl_colors[1])
    pGME = axs[2].plot(np.array(traces['sim_time']), ME_tot, label='$\mathrm{ME}_\mathrm{pert}$', color=Dark2_6_r.mpl_colors[2])
    KEb = axs[2].plot(np.array(traces['sim_time']), KE_background, label='$\mathrm{KE}_\mathrm{mean}$', color=Dark2_6_r.mpl_colors[3])
    KEgrowth=axs[2].plot(np.array(traces['sim_time']), KE_growth, label='$e^{2\mathrm{Re}[\lambda]t}$', color='darkgreen', linestyle='--')
    
    axs[2].text(-0.23, 1.15, r'$\mathrm{a)}$', horizontalalignment='center',
     verticalalignment='center', transform=axs[2].transAxes)
    axs[3].text(-0.23, 1.15, r'$\mathrm{c)}$', horizontalalignment='center',
     verticalalignment='center', transform=axs[3].transAxes)
    axs[4].text(-0.23, 1.15, r'$\mathrm{b)}$', horizontalalignment='center',
     verticalalignment='center', transform=axs[4].transAxes)
    axs[0].text(-0.23, 1.07, r'$\mathrm{d)}$', horizontalalignment='center',
     verticalalignment='center', transform=axs[0].transAxes)

    if add_energy_modes:
        axs[2].axhline(y=KE_modes, color='blue', linestyle='--', label='$\mathrm{KE}_\mathrm{linear modes}$')
        axs[2].axhline(y=ME_modes, color='darkorange', linestyle='--',label='$\mathrm{ME}_\mathrm{linear modes}$')
        KE_avg=scalar_time_avg(KE_tot, scalar_time)
        ME_avg=scalar_time_avg(ME_tot, scalar_time)
        energy_ratio=KE_avg/ME_avg
        linear_mode_ratio=KE_modes/ME_modes
        print('Non-linear energy ratio =', energy_ratio)
        print('Linear energy ratio =', linear_mode_ratio)
        print(KE_avg, ME_avg)
        
    rcParams.update({'font.size': 9})
    axs[2].legend(ncol=1, loc='lower center')
    rcParams.update({'font.size': 13})
    axs[2].set_xlabel('$\mathrm{Time}$')
    axs[2].set_ylim(5e-6,energy_ylim_max)
    axs[2].set_ylabel('$\mathrm{Energy}$')
    axs[2].set_yscale('log')
    axs[2].axvline(x=time, color='black', linestyle='--')
    axs[2].set_xlim(0, 100000)
    axs[3].plot(ks, disp_rel, color=Dark2_6_r.mpl_colors[4], label=r'$Re=100$''\n''$C_{B}=1$')
    axs[3].plot(ks, true_disp_rel, color=Dark2_6_r.mpl_colors[5], label=r'$Re=18.1$''\n''$C_{B}=30.7$')
    axs[3].set_ylim(0,disp_ylim_max)
    axs[3].set_xlim(0,disp_xlim_max)
    axs[3].axvline(x=kbox, color='black', linestyle='--')
    axs[3].set_xlabel('$k_{z}$')
    axs[3].set_ylabel('$\mathrm{Re}[\lambda]$')
    rcParams.update({'font.size': 9})
    axs[3].legend(loc='upper right')
    rcParams.update({'font.size': 13})

    axs[4].plot(z_profile, u_profile, color=Dark2_6_r.mpl_colors[2], label=r'$\langle u_{z} \rangle_{z,t}$')
    axs[4].plot(z_profile, fake_u_profile, color=Dark2_6_r.mpl_colors[0], linestyle='--', label=r'$U\sin(x)$')
    axs[4].set_xlabel('$x$')
    axs[4].set_ylabel('$u_z$')
    axs[4].set_xlim(0,4*np.pi)
    rcParams.update({'font.size': 9})
    axs[4].legend(loc='upper right')
    rcParams.update({'font.size': 13})

    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()


Re1=100
Pm1=0.1
Hb1=1
Lx1=170*np.pi
Lz1=4*np.pi
Nx1=256
Nz1=64
slices1='./slices_s340.h5'
traces1='./full_traces.h5'
profiles1='./averaged_avg_profs.h5'
dynamics_plot('func_pm0_1_re100_hb1.pdf', slices1, traces1, profiles1, 'u', 'w', r'$u_{z}$', r'$u_{x}$',\
              Lx1, Lz1, Nx1, Nz1, Re1, Hb1, Pm1,  1, 0.002, 0.1, 3e-4,0.04, (0.33, 0.42, 0.15, 0.2),\
              add_energy_modes=False )
