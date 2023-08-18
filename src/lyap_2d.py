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

def do_2d_lyapunov(game, seed, mix_coeff, gamma,
                    # Problem setup
                    # PUT ANY PARAMS FOR RR EXAMPLE HERE
                   alpha, opt, num_bin, ax_key, num_directions, do_sigmoid_range, num_lyapunov_iters,
                    save_name, title, do_print, do_legend, tune_first_dir, tune_every_dir,
                    use_smart_dir, do_trajectories
                  ):
    problem_setup = setup_game_and_opt(game, mix_coeff, gamma, opt, alpha, 
                                        num_lyapunov_iters, 
                                        use_fixed_direction=False,
                                        tune_first_dir=tune_first_dir, 
                                        tune_every_dir=tune_every_dir, 
                                        use_smart_dir=use_smart_dir,
                                        num_directions=num_directions, seed=seed)
    dims, Ls, grad_Ls, fixed_point_op, jac_fixed_point_op, update, get_lyapunov, get_entropy = problem_setup
    
        
    num_params = np.sum(dims)

    fig, axs = plt.subplots(1, 1)
    fig.tight_layout()
    fig.set_figheight(10), fig.set_figwidth(10)
    axs.set_aspect(aspect=1.0)

    if do_print:
        file = open("runs/" + save_name, "w")

    if do_sigmoid_range:
        eps = 1e-8
        x_ = logit(np.linspace(0.0 + eps, 1.0 - eps, num=num_bin))
        y_ = logit(np.linspace(0.0 + eps, 1.0 - eps, num=num_bin))
        @jax.jit
        def activation(x): return sigmoid(x)
    else:
        low_lim_x, up_lim_x = -8.0, 8.0
        low_lim_y, up_lim_y = low_lim_x, up_lim_x
        x_ = np.linspace(low_lim_x, up_lim_x, num=num_bin)
        y_ = np.linspace(low_lim_y, up_lim_y, num=num_bin)
        @jax.jit
        def activation(x): return x
    
    x,y = np.meshgrid(x_, y_)

    @jax.jit
    def get_prepped_losses(x, y):
        th = jnp.asarray([x, y])
        losses = Ls(th)
        return losses[0], losses[1]

    @jax.jit
    def get_prepped_grad(x, y):
        th = jnp.asarray([x, y])
        return grad_Ls(th)

    def get_prepped_lyapunov(x, y, seed):
        th = jnp.asarray([x, y])
        lyapunovs = get_lyapunov(th, seed)
        return lyapunovs

    def get_prepped_log_lyapunov(x, y, seed):
        th = jnp.asarray([x, y])
        lyapunovs = get_lyapunov(th, seed)
        return np.clip(np.log(np.abs(lyapunovs)), -9.0, 1.0)

    def get_prepped_entropy(x, y, seed):
        th = jnp.asarray([x, y])
        entropy = jnp.clip(get_entropy(th, seed), -10.0, 10.0)
        return entropy


    num_levels = 100
    # COLOR FOR GRAD
    if ax_key == 'grad':
        levels = np.array([[np.log(np.sqrt(np.sum(get_prepped_grad(x, y) ** 2))) for x in x_] for y in y_])
        c = axs.contourf(activation(x), activation(y), levels, num_levels)
        cbar = plt.colorbar(c, fraction=0.046, pad=0.04)
        cbar.set_label(r'Average Player gradient Norm', rotation=270, labelpad=25)
    elif ax_key == 'lyapunov':
        levels = np.array([[get_prepped_lyapunov(x, y, seed=ind1*100*100 + ind2) for ind1,
                            x in enumerate(x_)] for ind2, y in enumerate(y_)])
        c = axs.contourf(activation(x), activation(y), levels, num_levels)
        cbar = fig.colorbar(c, fraction=0.046, pad=0.04)
        cbar.set_label(r'Lyapunov exponent', rotation=270, labelpad=25)
    elif ax_key == 'log_lyapunov':
        levels = np.array([[get_prepped_log_lyapunov(x, y, seed=ind1*100*100 + ind2) for ind1, 
                            x in enumerate(x_)] for ind2, y in enumerate(y_)])
        c = axs.contourf(activation(x), activation(y), levels, num_levels)
        cbar = fig.colorbar(c, fraction=0.046, pad=0.04)
        cbar.set_label(r'log|Lyapunov exponent|', rotation=270, labelpad=25)
    elif ax_key == 'loss':
        levels = np.array([[np.mean(get_prepped_losses(x, y)[0]) for x in x_] for y in y_])
        c = axs.contourf(activation(x), activation(y), levels, num_levels)
        cbar = fig.colorbar(c, fraction=0.046, pad=0.04)
        cbar.set_label(r'Average Loss of players', rotation=270, labelpad=25)
    elif ax_key == 'entropy':
        levels = np.array([[get_prepped_entropy(x, y, seed=ind1*100*100 + ind2) for ind1, 
                            x in enumerate(x_)] for ind2, y in enumerate(y_)])
        c = axs.contourf(activation(x), activation(y), levels, num_levels)
        cbar = fig.colorbar(c, fraction=0.046, pad=0.04)
        cbar.set_label(r'Lyapunov Entropy', rotation=270, labelpad=25)
        

    if do_trajectories:
        color = 'r' #cm.rainbow(np.linspace(0, 1, len(x_)*len(y_)))
        i=0
        num_iters = num_lyapunov_iters
        #for i in tqdm(range(num_bin)):
        for x in x_:
            for y in y_:
                th_current = jnp.array([x, y])
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
                    ths.append(th_current)
                    lyap_current = get_lyapunov(th_current, seed=seed)
                    lyaps.append(lyap_current)
                    losses.append(jnp.abs(lyap_current))
                    if do_print: 
                        file.write(f"iter={_},det={determinants[-1]},trace={traces[-1]},lyap={lyaps[-1]},loss={losses[-1]},th={ths[-1]}\n") #th={ths[-1]}
                ths = np.array(ths)
                axs.plot(activation(ths[:, 0]), activation(ths[:, 1]), alpha=0.1, c=color)
                i += 1
        
    if do_legend:
        plt.legend(loc='upper left')

    if do_sigmoid_range:
       axs.set_xlim([0, 1])
       axs.set_xticks([0.0, 0.5, 1.0])
       axs.set_xticklabels(['0', '0.5', '1'])
       axs.set_ylim([0, 1])
       axs.set_yticks([0.0, 0.5, 1.0])
       axs.set_yticklabels(['0', '0.5', '1'])
    else:
       axs.set_xlim([low_lim_x*1.5, up_lim_x*1.5])
       axs.set_ylim([low_lim_y*1.5, up_lim_y*1.5])

    axs.set_xlabel("Player 1 P(D|State)")
    axs.set_ylabel("Player 2 P(D|State)")
    plt.title(title, pad=20)
    
    #image = open(, 'x')
    plt.savefig("images/" + save_name + '.pdf', transparent=True, bbox_inches='tight', dpi=300)
    if do_print: file.close()
