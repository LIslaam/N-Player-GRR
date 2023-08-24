import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from grr import setup_game_and_opt
from environments import *

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.family': 'serif'})

#jax.default_device = jax.devices("gpu")[0]

def do_3d_lyapunov(game, seed, mix_coeff, gamma,
                   alpha, opt, num_bin, ax_key, num_directions, do_sigmoid_range, num_lyapunov_iters,
                    save_name, title, do_print, do_legend, tune_first_dir, tune_every_dir,
                    use_smart_dir, do_trajectories):
  
    problem_setup = setup_game_and_opt(game, mix_coeff, gamma, opt, alpha, 
                                        num_lyapunov_iters, 
                                        use_fixed_direction=False,
                                        tune_first_dir=tune_first_dir, 
                                        tune_every_dir=tune_every_dir, 
                                        use_smart_dir=use_smart_dir,
                                        num_directions=num_directions, seed=seed)
    dims, Ls, grad_Ls, fixed_point_op, jac_fixed_point_op, update, get_lyapunov, get_entropy = problem_setup

    num_params = np.sum(dims)

    if do_print:
        file = open("runs/"+save_name+".txt", "w")
    
    if do_sigmoid_range:
        eps = 1e-8
        x_ = logit(np.linspace(0.0 + eps, 1.0 - eps, num=num_bin)) # From -18 to +18 approx
        y_ = logit(np.linspace(0.0 + eps, 1.0 - eps, num=num_bin)) # activation gives 1e-8 to approx 1
        z_ = logit(np.linspace(0.0 + eps, 1.0 - eps, num=num_bin))
        def activation(x): return sigmoid(x) # Checked this section is correct.
    else:
        low_lim_x, up_lim_x = -8.0, 8.0
        low_lim_y, up_lim_y = low_lim_x, up_lim_x
        low_lim_z, up_lim_z = low_lim_x, up_lim_x
        x_ = np.linspace(low_lim_x, up_lim_x, num=num_bin)
        y_ = np.linspace(low_lim_y, up_lim_y, num=num_bin)
        z_ = np.linspace(low_lim_z, up_lim_z, num=num_bin)
        def activation(x): return x
  
    x_xy, y_xy = np.meshgrid(x_, y_)
    y_yz, z_yz = np.meshgrid(y_, z_)
    x_xz, z_xz = np.meshgrid(x_, z_)

    @jax.jit
    def get_prepped_losses(x, y, z):
        th = jnp.asarray([x, y, z])
        losses = Ls(th)
        return losses[0], losses[1], losses[2]

    @jax.jit
    def get_prepped_grad(x, y,z):
        th = jnp.asarray([x, y, z])
        return grad_Ls(th)


    def get_prepped_lyapunov(x, y, z, seed):
        th = jnp.asarray([x, y, z])
        lyapunovs = get_lyapunov(th, seed)
        return lyapunovs

    def get_prepped_log_lyapunov(x, y, z, seed):
        th = jnp.asarray([x, y, z])
        lyapunovs = get_lyapunov(th, seed)
        return np.clip(np.log(np.abs(lyapunovs)), -9.0, 1.0)

    def get_prepped_entropy(x, y, z, seed):
        th = jnp.asarray([x, y, z])
        entropy = jnp.clip(get_entropy(th, seed), -10.0, 10.0)
        return entropy

    # --- Make 3D figure -----
    fig = plt.figure()
    axs = plt.axes(projection='3d')
    fig.tight_layout()
    fig.set_figheight(10), fig.set_figwidth(10)
    # --- 3 2D figures views of surface of cube ------
    figxy, axsxy = plt.subplots(1, 1)
    figxy.tight_layout()
    figxy.set_figheight(10), figxy.set_figwidth(10)
    axsxy.set_aspect(aspect=1.0)

    figyz, axsyz = plt.subplots(1, 1)
    figyz.tight_layout()
    figyz.set_figheight(10), figyz.set_figwidth(10)
    axsyz.set_aspect(aspect=1.0)

    figxz, axsxz = plt.subplots(1, 1)
    figxz.tight_layout()
    figxz.set_figheight(10), figxz.set_figwidth(10)
    axsxz.set_aspect(aspect=1.0)

    num_levels = 100
    if ax_key == 'lyapunov':
        levels = np.array([[[get_prepped_lyapunov(x, y, z, seed=ind1*100*100 + ind2 + ind3) for ind1,
                            x in enumerate(x_)] for ind2, y in enumerate(y_)] for ind3, z in enumerate(z_)])
        # Reshaping and rotating levels for plotting yz, xz
        levels_yz = [levels[n][0, :num_bin] for n in range(num_bin)]
        levels_yz = [[levels_yz[i][j] for i in range(num_bin)] for j in range(num_bin)] # Rotating array
        
        levels_xz = [levels[n][:num_bin, 0] for n in range(num_bin)]
        levels_xz = [[levels_xz[i][j] for i in range(num_bin)] for j in range(num_bin)] # Rotating array

        cxy = axsxy.contourf(activation(x_xy), activation(y_xy), levels[0], num_levels, alpha=0.5)
        cyz = axsyz.contourf(activation(y_yz), activation(z_yz), levels_yz, num_levels, alpha=0.5)
        cxz = axsxz.contourf(activation(x_xz), activation(z_xz), levels_xz, num_levels, alpha=0.5)

        cbarxy = fig.colorbar(cxy, fraction=0.046, pad=0.04)
        cbarxy.set_label(r'Lyapunov exponent', rotation=270, labelpad=25)
        cbaryz = fig.colorbar(cyz, fraction=0.046, pad=0.04)
        cbaryz.set_label(r'Lyapunov exponent', rotation=270, labelpad=25)
        cbarxz = fig.colorbar(cxz, fraction=0.046, pad=0.04)
        cbarxz.set_label(r'Lyapunov exponent', rotation=270, labelpad=25)

    if do_trajectories:
        color = 'r' #cm.rainbow(np.linspace(0, 1, len(x_)*len(y_)))
        num_iters = num_lyapunov_iters
        for x in x_:
            for y in y_:
                for z in z_:
                    th_current = jnp.array([x, y, z])
                    ths = [th_current]
                    jacobian = jac_fixed_point_op(th_current)
                    traces = [jnp.trace(jacobian)]
                    determinants = [jnp.linalg.det(jacobian)]
                    lyap_current = get_lyapunov(th_current, seed=0)
                    lyaps = [lyap_current]
                    losses = [jnp.abs(lyap_current)]
                    for _ in range(num_iters):
                        th_current = fixed_point_op(th_current)
                        jacobian = jac_fixed_point_op(th_current)
                        traces.append(jnp.trace(jacobian))
                        determinants.append(jnp.linalg.det(jacobian))
                        ths.append(th_current)
                        lyap_current = get_lyapunov(th_current, seed=seed)
                        lyaps.append(lyap_current)
                        losses.append(jnp.abs(lyap_current))
                        if do_print: 
                            file.write(f"iter={_},det={determinants[-1]},trace={traces[-1]},lyap={lyaps[-1]},loss={losses[-1]},th={ths[-1]}\n") #th={ths[-1]}
                    ths = np.array(ths)
                    # Plotting the trajectory!
                    axs.plot(activation(ths[:, 0]), activation(ths[:, 1]), activation(ths[:, 2]), alpha=0.1, color=color)
                    axsxy.plot(activation(ths[:, 0]), activation(ths[:, 1]), alpha=0.1, color=color)
                    axsyz.plot(activation(ths[:, 1]), activation(ths[:, 2]), alpha=0.1, color=color)
                    axsxz.plot(activation(ths[:, 0]), activation(ths[:, 2]), alpha=0.1, color=color)

    if do_legend:
        fig.legend(loc='upper left')

    if do_sigmoid_range:
        axs.set_xlim([0, 1.])
        axs.set_xticks([0.0, 0.5, 1.0])
        axs.set_xticklabels(['0', '0.5', '1'])
        axs.set_ylim([0, 1.])
        axs.set_yticks([0.0, 0.5, 1.0])
        axs.set_yticklabels(['0', '0.5', '1'])
        axs.set_zlim([0, 1.])
        axs.set_zticks([0.0, 0.5, 1.0])
        axs.set_zticklabels(['0', '0.5', '1'])

        axsxy.set_xlim([0, 1.])
        axsxy.set_xticks([0.0, 0.5, 1.0])
        axsxy.set_xticklabels(['0', '0.5', '1'])
        axsxy.set_ylim([0, 1.])
        axsxy.set_yticks([0.0, 0.5, 1.0])
        axsxy.set_yticklabels(['0', '0.5', '1'])

        axsyz.set_xlim([0, 1.])
        axsyz.set_xticks([0.0, 0.5, 1.0])
        axsyz.set_xticklabels(['0', '0.5', '1'])
        axsyz.set_ylim([0, 1.])
        axsyz.set_yticks([0.0, 0.5, 1.0])
        axsyz.set_yticklabels(['0', '0.5', '1'])

        axsxz.set_xlim([0, 1.])
        axsxz.set_xticks([0.0, 0.5, 1.0])
        axsxz.set_xticklabels(['0', '0.5', '1'])
        axsxz.set_ylim([0, 1.])
        axsxz.set_yticks([0.0, 0.5, 1.0])
        axsxz.set_yticklabels(['0', '0.5', '1'])

    else: # Didn't fix for 3D
        axs.set_xlim([low_lim_x*1.5, up_lim_x*1.5])
        axs.set_ylim([low_lim_y*1.5, up_lim_y*1.5])
        axs.set_zlim([low_lim_z*1.5, up_lim_z*1.5])
    axs.set_xlabel("Player 1 P(D|State)") # 'Defence', or 'Defect' depending on game
    axs.set_ylabel("Player 2 P(D|State)")
    axs.set_zlabel("Player 3 P(D|State)")

    axsxy.set_xlabel("Player 1 P(D|State)") # 'Defence', or 'Defect' depending on game
    axsxy.set_ylabel("Player 2 P(D|State)")
    axsyz.set_xlabel("Player 2 P(D|State)") # 'Defence', or 'Defect' depending on game
    axsyz.set_ylabel("Player 3 P(D|State)")
    axsxz.set_xlabel("Player 1 P(D|State)") # 'Defence', or 'Defect' depending on game
    axsxz.set_ylabel("Player 3 P(D|State)")
    axs.set_title(title, pad=20)
    axsxy.set_title(title+' XY View', pad=20)
    axsyz.set_title(title+' YZ View', pad=20)
    axsxz.set_title(title+' XZ View', pad=20)
  
    fig.savefig(f'{"images/"+save_name+"_3D"}.pdf', 
                transparent=True, bbox_inches='tight', dpi=300)
    figxy.savefig(f'{"images/"+save_name+"_XY"}.pdf', 
                transparent=True, bbox_inches='tight', dpi=300)
    figyz.savefig(f'{"images/"+save_name+"_YZ"}.pdf', 
                transparent=True, bbox_inches='tight', dpi=300)
    figxz.savefig(f'{"images/"+save_name+"_XZ"}.pdf', 
                transparent=True, bbox_inches='tight', dpi=300)
    if do_print: file.close()