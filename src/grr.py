import pkg_resources
#pkg_resources.require("jax==0.2.22")
#pkg_resources.require("jaxlib==0.1.76")
import jax
from environments import *
from optimisers import * 

#jax.default_device = jax.devices("gpu")[0]


def setup_game_and_opt(game, mix_coeff, gamma, opt, alpha, num_lyapunov_iters, use_fixed_direction,
                       tune_first_dir, tune_every_dir, use_smart_dir, num_directions, seed,
                       entropy_objective_strat='min'):
  if game == 'mix':
    dims, Ls, grad_Ls, = matching_and_ipd(gamma=gamma, fixed_defect=-3.0, weighting=mix_coeff)
  elif game == 'mp':
    dims, Ls, grad_Ls, = matching_and_ipd(gamma=0.9, fixed_defect=-3.0, weighting=1.0)
  elif game == 'small-ipd':
    dims, Ls, grad_Ls, = matching_and_ipd(gamma=0.9, fixed_defect=-3.0, weighting=0.0)
  elif game == 'random-ipd':
    dims, Ls, grad_Ls, = ipd_random_proj(gamma=0.96, seed=seed)
  # elif game == 'rr':
  #   dims, Ls, grad_Ls = rr_example()
  # elif game == 'henon':
  #   dims, Ls, grad_Ls = henon_example()
  elif game == 'ipd':
    dims, Ls, grad_Ls = ipd()
  else:
    assert False, "Provide a valid game choice"
  if opt == 'sgd':
    fixed_point_op, jac_fixed_point_op, update = simul_sgd(grad_Ls, alpha=alpha)
  elif opt == 'eg':
    fixed_point_op, jac_fixed_point_op, update = simul_lola(grad_Ls, alpha=alpha)
  else:
    assert False, "Provide a valid optimizer choice"

  @jax.jit 
  def get_random_direction(key):
    key, subkey = jax.random.split(key)
    direction = jax.random.uniform(key=subkey, shape=(dims[0] + dims[1],), 
                                   minval=-1.0, maxval=1.0)
    direction = direction / jnp.linalg.norm(direction)
    return direction, key


  def do_lyap_term(th, direction, maximize_direction):
    J = jac_fixed_point_op(th) 
    M = J @ J.T
    if maximize_direction:
      for _ in range(10):
          pre_direction = M @ direction
          direction = pre_direction / jnp.linalg.norm(pre_direction)
    new_dir = direction
    pre_term = new_dir.T @ M @ new_dir

    return fixed_point_op(th), pre_term, new_dir / jnp.linalg.norm(direction)


  def get_lyapunov_full(th, tune_first_dir, tune_every_dir, use_smart_dir, 
                        seed=0):
    key = jax.random.PRNGKey(seed)
    result = 0.0
    if use_smart_dir: 
      direction = get_smart_direction(th, seed)
    else:
      direction, key = get_random_direction(key)
    for t in range(num_lyapunov_iters):
      if t == 0 and tune_first_dir is True:
        maximize_direction = True
      elif tune_every_dir is True:
        maximize_direction = True
      else:
        maximize_direction = False
      th_old = th
      th, pre_term, direction = do_lyap_term(th, direction, maximize_direction)

      result += jnp.log(jnp.abs(pre_term))
    return (1.0/num_lyapunov_iters) * result, direction


  def get_smart_direction(th, seed):
    return get_lyapunov_full(th, tune_first_dir=False, tune_every_dir=True, 
                             use_smart_dir=False, seed=seed)[1]


  def get_lyapunov(th, seed=0):
    return get_lyapunov_full(th, tune_first_dir, tune_every_dir, 
                             use_smart_dir, seed)[0]


  def get_random_directions(key):
    key, subkey = jax.random.split(key)
    directions = jax.random.uniform(key=subkey, shape=(num_directions, dims[0] + dims[1]), minval=-1.0, maxval=1.0)
    directions = jnp.asarray([direction / jnp.linalg.norm(direction) 
                            for direction in directions])
    return directions, key


  def do_entropy_term(th, directions, maximize_directions):
    J = jac_fixed_point_op(th) 
    # DO POWER ITERATION
    M = J @ J.T
    w, v = jax.numpy.linalg.eigh(M)
    sorted_args = jnp.argsort(jnp.real(w))[::-1]
    pre_terms = jnp.asarray([jnp.real(w[sorted_arg]) for sorted_arg in sorted_args[:num_directions]])

    return fixed_point_op(th), pre_terms, directions


  def get_entropy_full(th, seed=0):
    key = jax.random.PRNGKey(seed)
    result_terms = 0.0
    directions, key = get_random_directions(key)
    num_entropy_iters = num_lyapunov_iters
    for t in range(num_entropy_iters):
      if t == 0 and tune_first_dir is True:
        maximize_directions = True
      elif tune_every_dir is True:
        maximize_directions = True
      else:
        maximize_directions = False
      th_old = th
      th, pre_terms, directions = do_entropy_term(th, directions, maximize_directions)
      result_terms += jnp.log(jnp.abs(pre_terms))
    result = 0.0

    # THIS IS AVERAGE SUMMARY
    # for result_term in result_terms:
    #   #if result_term > 0.0:  # FOR GREATER THAN 0 SUMMARY (i.e., ENTROPY)
    #   result += (1.0/num_lyapunov_iters) * result_term

    # THIS IS MIN SUMMARY
    if entropy_objective_strat == 'min':
      result = jnp.min(result_terms)
    elif entropy_objective_strat == 'sum':
      for result_term in result_terms:
        result += result_term
    elif entropy_objective_strat == 'entropy':
      for result_term in result_terms:
        if result_term > 0.0:  # FOR GREATER THAN 0 SUMMARY (i.e., ENTROPY)
          result += result_term
    return result, directions

  def get_entropy(th, seed=0):
    return get_entropy_full(th, seed)[0]

  return dims, Ls, grad_Ls, fixed_point_op, jac_fixed_point_op, update, \
  get_lyapunov, get_entropy