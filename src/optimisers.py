#import pkg_resources
#pkg_resources.require("jax==0.2.22")
#pkg_resources.require("jaxlib==0.1.76")
import jax 

def simul_sgd(grad_func, alpha=0.025):
  @jax.jit
  def update(th):
    return alpha*grad_func(th)

  @jax.jit
  def fixed_point_op(th):
    #print(th)
    #print(update(th))
    return th - update(th)

  #jac_fixed_point_op_func = jax.jit(jax.jacfwd(fixed_point_op, argnums=(0)))
  jac_fixed_point_op_func = jax.jit(jax.jacfwd(fixed_point_op))
  @jax.jit
  def jac_fixed_point_op(th):
    # TODO: FIX THIS, BUT FIXING SEED to 0 FOR NOW, SINCE WE NEVER DO LYAP OF LYAP?
    return jac_fixed_point_op_func(th)

  return fixed_point_op, jac_fixed_point_op, update


def simul_lola(grad_func, alpha=0.025):
  @jax.jit
  def update(th):
    g_temp = grad_func(th)
    return alpha*grad_func(th - alpha*g_temp)

  @jax.jit
  def fixed_point_op(th):
    return th - update(th)

  jac_fixed_point_op_func = jax.jit(jax.jacfwd(fixed_point_op))
  @jax.jit
  def jac_fixed_point_op(th):
    return jac_fixed_point_op_func(th)

  return fixed_point_op, jac_fixed_point_op, update