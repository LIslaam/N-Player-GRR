import pkg_resources
pkg_resources.require("jax==0.2.22")
pkg_resources.require("jaxlib==0.1.76")
import jax
import jax.numpy as jnp

@jax.jit
def sigmoid(x): return 1 / (1 + jnp.exp(-x))

@jax.jit
def activation(x): return sigmoid(x)

def logit(x): return jnp.log(x / (1.0 - x)) 


#------------ AN ENVIRONMENT LABEEBAH WROTE ---------
def offense_defense(gamma=0.99, fixed_defect=-3.0): # Fixed defect: We fix the strategy if our
                                                        #opponent defects, to defect with high probability
  dims = [1, 1, 1] 
  payout_mat_1 = jnp.array([[[0,1],[-2,1]], [[1,-2],[1,0]]]) # 3D tensor
  payout_mat_2 = jnp.array([[[0,-2],[1,1]], [[1,1],[-2,0]]])
  payout_mat_3 = jnp.array([[[0,1],[1,-2]], [[-2,1],[1,0]]]) # 3 players, 3 matrices

  @jax.jit
  def get_M(th):
    p_1_0, p_2_0, p_3_0  = (jnp.array([activation(th[0])]), 
                            jnp.array([activation(th[1])]),
                            jnp.array([activation(th[2])]))
    p = jnp.concatenate([p_1_0*p_2_0*p_3_0, p_1_0*p_2_0*(1-p_3_0), 
                         p_1_0*(1-p_2_0)*p_3_0, (1-p_1_0)*p_2_0*p_3_0,
                         (1-p_1_0)*p_2_0*(1-p_3_0), (1-p_1_0)*(1-p_2_0)*p_3_0,
                         p_1_0*(1-p_2_0)*(1-p_3_0), (1-p_1_0)*(1-p_2_0)*(1-p_3_0)])
    
    new_p_1, new_p_2, new_p_3 = (jnp.reshape(activation(th[0]), (1, 1)), 
                                 jnp.reshape(activation(th[1]), (1, 1)),
                                 jnp.reshape(activation(th[1]), (1, 1)))
    fixed = jnp.reshape(activation(jnp.array([fixed_defect])), (1, 1)) # fixed high probability of offense if opponent offends

    #                               P(D|DDD), P(D|DDO), P(D|DOD), P(D|ODD), P(D|DOO), P(D|ODO), P(D|OOD), P(D|OOO)
    modified_th0 = jnp.concatenate([new_p_1, fixed, fixed, new_p_1, fixed, fixed, fixed, fixed]) # jnp.concatenate([fixed, fixed, fixed, fixed, fixed, fixed, fixed, fixed]) # jnp.concatenate([new_p_1, new_p_1, new_p_1, new_p_1, fixed, new_p_1, new_p_1, fixed]) #  # CAT LINE
    modified_th1 = jnp.concatenate([new_p_2, fixed, new_p_2, fixed, fixed, fixed, fixed, fixed]) # jnp.concatenate([fixed, fixed, fixed, fixed, fixed, fixed, fixed, fixed]) # jnp.concatenate([new_p_2, new_p_2, new_p_2, new_p_2, new_p_2, fixed, new_p_2, fixed])  # CAT LINE   
    modified_th2 = jnp.concatenate([fixed, fixed, fixed, fixed, fixed, fixed, fixed, fixed]) # jnp.concatenate([new_p_3, new_p_3, new_p_3, new_p_3, new_p_3, new_p_3, fixed, fixed]) # jnp.concatenate([new_p_3, new_p_3, fixed, fixed, fixed, fixed, fixed, fixed]) # CAT LINE #Freezing agent! Not learning #
    p_1, p_2, p_3 = (jnp.reshape(modified_th0, (8, 1)), 
                     jnp.reshape(modified_th1, (8, 1)),
                     jnp.reshape(modified_th2, (8, 1)))# 8 states
    P = jnp.concatenate([p_1*p_2*p_3, p_1*p_2*(1-p_3), 
                         p_1*(1-p_2)*p_3, (1-p_1)*p_2*p_3,
                         (1-p_1)*p_2*(1-p_3), (1-p_1)*(1-p_2)*p_3,
                         p_1*(1-p_2)*(1-p_3), (1-p_1)*(1-p_2)*(1-p_3)],
                        axis=1)  # CAT LINE
    M = -jnp.matmul(p, jnp.linalg.inv(jnp.eye(8)-gamma*P))

    return M

  @jax.jit
  def L_1(th):
    M = get_M(th)
    L_1 = jnp.matmul(M, jnp.reshape(payout_mat_1, (8, 1)))
    return (1 - gamma)*L_1[0]

  @jax.jit
  def L_2(th):
    M = get_M(th)
    L_2 = jnp.matmul(M, jnp.reshape(payout_mat_2, (8, 1)))
    return (1 - gamma)*L_2[0]

  @jax.jit
  def L_3(th):
    M = get_M(th)
    L_3 = jnp.matmul(M, jnp.reshape(payout_mat_3, (8, 1)))
    return (1 - gamma)*L_3[0]

  @jax.jit 
  def Ls(th): return jnp.asarray([L_1(th), L_2(th), L_3(th)])

  grad_L_1 = jax.jit(jax.grad(L_1, argnums=(0)))
  grad_L_2 = jax.jit(jax.grad(L_2, argnums=(0)))
  grad_L_3 = jax.jit(jax.grad(L_3, argnums=(0)))
  @jax.jit
  def grad_Ls(th): return jnp.asarray([grad_L_1(th)[0], 
                                           grad_L_2(th)[1],
                                           grad_L_3(th)[2]])

  return dims, Ls, grad_Ls, # grad_L_1, grad_L_2, grad_L_3# get_M, L_1
# ------- ^^^^ The 3 player environment Labeebah wrote

  
def matching_and_ipd(gamma=0.99, fixed_defect=-3.0, weighting=.25):
  dims = [1, 1]
  payout_mat_1_matching = jnp.array([[1,-1],[-1,1]])
  payout_mat_2_matching = -payout_mat_1_matching
  payout_mat_1_ipd = jnp.array([[-1,-3],[0,-2]])
  payout_mat_2_ipd = payout_mat_1_ipd.T

  @jax.jit
  def get_M(th):
    p_1_0, p_2_0  = jnp.array([activation(th[0])]), jnp.array([activation(th[1])])
    p = jnp.concatenate([p_1_0*p_2_0, p_1_0*(1-p_2_0), (1-p_1_0)*p_2_0, 
                         (1-p_1_0)*(1-p_2_0)])  # CAT LINE
    new_p_1, new_p_2 = jnp.reshape(activation(th[0]), (1, 1)), jnp.reshape(activation(th[1]), (1, 1))
    fixed = jnp.reshape(activation(jnp.array([fixed_defect])), (1, 1))

    modified_th0 = jnp.concatenate([new_p_1, fixed, new_p_1, fixed]) # CAT LINE
    modified_th1 = jnp.concatenate([new_p_2, new_p_2, fixed, fixed]) # CAT LINE
    p_1, p_2 = jnp.reshape(modified_th0, (4, 1)), jnp.reshape(modified_th1, (4, 1))
    P = jnp.concatenate([p_1*p_2, p_1*(1-p_2), (1-p_1)*p_2, (1-p_1)*(1-p_2)],
                        axis=1)  # CAT LINE
    M = -jnp.matmul(p, jnp.linalg.inv(jnp.eye(4)-gamma*P))

    return M

  @jax.jit
  def L_1(th):
    p_1_match, p_2_match = jnp.array([sigmoid(th[0])]), jnp.array([sigmoid(th[1])])
    x, y = jnp.concatenate([p_1_match, 1-p_1_match]), jnp.concatenate([p_2_match, 1-p_2_match]) # CAT LINE
    L_1_matching = jnp.matmul(jnp.matmul(x, payout_mat_1_matching), y)

    M = get_M(th)
    L_1_ipd = jnp.matmul(M, jnp.reshape(payout_mat_1_ipd, (4, 1)))
    L = (weighting)*L_1_matching + (1 - gamma)*(1 - weighting)*L_1_ipd

    return L[0]
    
  @jax.jit
  def L_2(th):
    p_1_match, p_2_match = jnp.array([sigmoid(th[0])]), jnp.array([sigmoid(th[1])])
    x, y = jnp.concatenate([p_1_match, 1-p_1_match]), jnp.concatenate([p_2_match, 1-p_2_match])
    L_2_matching = jnp.matmul(jnp.matmul(x, payout_mat_2_matching), y)

    M = get_M(th)
    L_2_ipd = jnp.matmul(M, jnp.reshape(payout_mat_2_ipd, (4, 1)))

    L = (weighting)*L_2_matching + (1 - gamma)*(1 - weighting)*L_2_ipd
    return L[0]
  
  @jax.jit
  def Ls(th): return [L_1(th), L_2(th)]

  grad_L_1 = jax.jit(jax.grad(L_1, argnums=(0)))
  grad_L_2 = jax.jit(jax.grad(L_2, argnums=(0)))
  @jax.jit
  def grad_Ls(th): return jnp.asarray([grad_L_1(th)[0], grad_L_2(th)[1]])
    
  return dims, Ls, grad_Ls

def ipd(gamma=0.96):
  dims = [5, 5]
  payout_mat_1 = jnp.array([[-1,-3],[0,-2]])
  payout_mat_2 = payout_mat_1.T

  @jax.jit
  def get_M(th):
    p_1_0 = jnp.reshape(activation(th[0]), (1, 1))
    p_2_0 = jnp.reshape(activation(th[dims[0]]), (1, 1))
    p = jnp.concatenate([p_1_0*p_2_0, p_1_0*(1-p_2_0), (1-p_1_0)*p_2_0, 
                         (1-p_1_0)*(1-p_2_0)]).reshape(1, 4)
    p_1 = jnp.reshape(activation(th[1:dims[0]]), (4, 1))
    p_2 = jnp.reshape(activation(th[dims[0] + 1:]), (4, 1))
    P = jnp.concatenate([p_1*p_2, p_1*(1-p_2), (1-p_1)*p_2,
                         (1-p_1)*(1-p_2)], axis=1)
    M = -jnp.matmul(p, jnp.linalg.inv(jnp.eye(4)-gamma*P))

    return M

  @jax.jit
  def L_1(th):
    M = get_M(th)
    L_1 = jnp.matmul(M, jnp.reshape(payout_mat_1, (4, 1)))
    return (1 - gamma)*L_1[0][0]

  @jax.jit
  def L_2(th):
    M = get_M(th)
    L_2 = jnp.matmul(M, jnp.reshape(payout_mat_2, (4, 1)))
    return (1 - gamma)*L_2[0][0]

  @jax.jit 
  def Ls(th): return jnp.asarray([L_1(th), L_2(th)])

  grad_L_1 = jax.jit(jax.grad(L_1, argnums=(0)))
  grad_L_2 = jax.jit(jax.grad(L_2, argnums=(0)))
  @jax.jit
  def grad_Ls(th): return jnp.concatenate([grad_L_1(th)[:dims[0]], 
                                           grad_L_2(th)[dims[0]:]])

  return dims, Ls, grad_Ls


def ipd_random_proj(gamma=0.96, seed=0, reg=0.0):
  dims = [1, 1]
  payout_mat_1 = jnp.array([[-1,-3],[0,-2]])
  payout_mat_2 = payout_mat_1.T
  np.random.seed(seed)
  proj_direction = np.random.uniform(size=(10,2), low=-2.0, high=2.0)
  proj_direction[5:,0] = 0.0
  proj_direction[:5,1] = 0.0
  proj_direction = jnp.asarray(proj_direction)
  proj_bias = logit(jnp.asarray([0.5, 0.75, 0.25, 0.75, 0.25, 0.5, 0.75, 0.75,
                                 0.25, 0.25])).reshape(10,)

  @jax.jit
  def get_M(th):
    th_full = proj_direction @ th + proj_bias
    p_1_0 = jnp.reshape(activation(th_full[0]), (1, 1))
    p_2_0 = jnp.reshape(activation(th_full[5]), (1, 1))
    p = jnp.concatenate([p_1_0*p_2_0, p_1_0*(1-p_2_0), (1-p_1_0)*p_2_0, (1-p_1_0)*(1-p_2_0)]).reshape(1, 4)
    p_1 = jnp.reshape(activation(th_full[1:5]), (4, 1))
    p_2 = jnp.reshape(activation(th_full[5 + 1:]), (4, 1))
    P = jnp.concatenate([p_1*p_2, p_1*(1-p_2), (1-p_1)*p_2, (1-p_1)*(1-p_2)], axis=1)
    M = -jnp.matmul(p, jnp.linalg.inv(jnp.eye(4)-gamma*P))

    return M


  @jax.jit
  def L_1(th):
    M = get_M(th)
    L_1 = jnp.matmul(M, jnp.reshape(payout_mat_1, (4, 1)))
    return (1 - gamma)*L_1[0][0] + reg*jnp.sum(th**2)


  @jax.jit
  def L_2(th):
    M = get_M(th)
    L_2 = jnp.matmul(M, jnp.reshape(payout_mat_2, (4, 1)))
    return (1 - gamma)*L_2[0][0] + + reg*jnp.sum(th**2)


  @jax.jit 
  def Ls(th): return jnp.asarray([L_1(th), L_2(th)])

  grad_L_1 = jax.jit(jax.grad(L_1, argnums=(0)))
  grad_L_2 = jax.jit(jax.grad(L_2, argnums=(0)))
  @jax.jit
  def grad_Ls(th):
    return jnp.concatenate([grad_L_1(th)[:dims[0]], grad_L_2(th)[dims[0]:]])

  return dims, Ls, grad_Ls