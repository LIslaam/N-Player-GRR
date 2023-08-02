import pkg_resources
pkg_resources.require("jax==0.2.22")
pkg_resources.require("jaxlib==0.1.76")
import os
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from lyap_2d import do_2d_lyapunov

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--optimiser", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_bin", type=int, default=11)
parser.add_argument("--print", type=bool, default=True)
args = parser.parse_args()


if __name__ == "__main__":
    ############################################
    seed = args.seed
    num_bin = args.num_bin
    do_print = args.print
    game = args.game
    opt = args.optimiser
    
    mix_coeff = 0.25
    gamma = 0.9
    alpha = 1.0
    ax_key = 'grad'
    d_strategy = 'do_ub'
    num_directions = 1
    num_lyapunov_iters = 10

    do_sigmoid_range = True # Range between 0 and 1
    do_legend = False
    tune_first_dir=True
    tune_every_dir=False
    use_smart_dir=False
    do_trajectories=True

    if game == 'small-ipd':
        game_labels = 'Small IPD'
    elif game == 'mix':
        game_labels = 'Mixed Small IPD and MP'
    elif game == 'mp':
        game_labels = 'Matching Pennies'

    if opt == 'sgd':
        opt_labels = 'SGD'
    elif opt == 'lola':
        opt_labels = 'LOLA'
    

    print(f"Doing game = {game}")
    print(f"\tDoing opt = {opt}")
    print(f"\t\t\tDoing ax key = {ax_key}")
    if ax_key == 'lyapunov' or ax_key == 'log_lyapunov':
        print(f"\t\t\t\tDoing num_lyapunov_iters = {num_lyapunov_iters}")

    if d_strategy == 'do_lb':
        tune_first_dir,  tune_every_dir, use_smart_dir = True, False, False
    elif d_strategy == 'do_ub':
        tune_first_dir,  tune_every_dir, use_smart_dir = False, True, False
    else: assert False, 'Use a valid d_strategy'

    print(f"\t\t\t\t\t d_stratregy = {d_strategy}")
    save_name = 'test_thetas_2d' # f'{game_labels[i]} {opt_labels[j]} $lr=${alpha}' #{alpha}' #{num_lyapunov_iters}-step Trajectories'
    #f"Game={game}_Opt={opt}_SigRange={do_sigmoid_range}_Key={temp_ax_key}_Num={num_lyapunov_iters}_dstrat={d_strategy}"
    title = f'{game_labels} {opt_labels} $lr=${alpha}'

    levels = do_2d_lyapunov(game, seed, mix_coeff, gamma, alpha, opt, num_bin, ax_key, num_directions,
                            do_sigmoid_range, num_lyapunov_iters, save_name, title, do_print, do_legend,
                            tune_first_dir, tune_every_dir, use_smart_dir, do_trajectories)
        