import numpy as np
from scipy import optimize
from multiprocess import Pool, freeze_support
import sympy
from sympy import symbols
from sympy import lambdify
import time

###############################################################################
#    Parameters                                                               
###############################################################################

n = 15
simulations = 10**6
batchnum = 5
seed = 5076377
pool_workers = 9
cost_tolerance = 10**(-10)
dup_tolerance = 10**(-4)

###############################################################################
#    Functions                                                             
###############################################################################

def cost_function(n):
    N = np.floor(n/2)
    x = symbols(f'x:{n}')
    sym_cost = n*N
    for k in np.arange(1,N+1):
        for j in np.arange(1,n):
            for i in np.arange(j):
                sym_cost += 2*sympy.cos(2*sympy.pi*(x[i]-x[int((i+k)%n)]-x[j]+x[int((j+k)%n)]))
    sym_cost = sym_cost.subs(x[0],0)
    cost_function = lambdify([x[1:]], sym_cost, 'numpy')
    sym_grad = [sym_cost.diff(xi) for xi in x[1:]]
    grad = lambdify([x[1:]], sym_grad, 'numpy')
    return (cost_function,grad)


def cazac_compute(cost,grad, tol, n, seed, step):
    new_seed = int((seed*(step+1))%(2**32))
    np.random.seed(new_seed)
    (x, val, d) = optimize.fmin_l_bfgs_b(cost, x0=np.random.rand(n-1), fprime=grad, factr=10.0)
    if val < tol:
        return x
    else:
        return None
    
def cazac_filter(arr, tol=10**(-3), max_iter=10000):
    index_len = 10000
    iters = 0
    out = np.ones((1,len(arr[0,:])),dtype=complex)
    remain = arr[:,:]
    while index_len > 0 and iters < max_iter:
        index = np.where(np.sum(np.abs(remain-remain[0,:])>tol,axis=1))[0]
        if len(index) > 0:
            out = np.vstack((out,remain[0,:]))
            remain = remain[index,:]
            iters += 1
        else:
            out = np.vstack((out,remain[0,:]))
            index_len = 0
    return out[1:,:]

def time_string(t):
    if t > 3600:
        hrs = int(t/3600)
        mins = int((t%3600)/60)
        secs = round(t%60,1)
        return f'[{hrs}h {mins}m {secs}s]'
    elif t > 60:
        mins = int(t/60)
        secs = round(t%60,1)
        return f'[{mins}m {secs}s]'
    else:
        return f'[{round(t,1)}s]'
    
###############################################################################
#    Main Script                                                            
###############################################################################
if __name__ == '__main__':
    freeze_support()
    t0 = time.time()
    filename = f'phase_cazac_length{n}_sim{simulations}_batch{batchnum}.csv'
    print(f'Computing cost funciton and gradient...', end='\r')
    cost, grad = cost_function(n)
    t1 = time.time()
    cost_time = time_string(t1-t0)
    print('Computing cost funciton and gradient...Done! '+cost_time)
    
    # Multiprocess CAZAC search
    t0 = time.time()
    print('Searching for CAZAC sequences...', end='\r')
    cazac_compute_step = lambda a: cazac_compute(cost,grad, cost_tolerance, n, seed, a)
    with Pool(pool_workers) as pool:
        cazac_list = pool.map(cazac_compute_step, range(simulations))
    cazac = []
    for i in np.arange(simulations):
        try:
            a = cazac_list[i][0]
            cazac.append(cazac_list[i])
        except:
            pass
    cazacs = np.modf(np.array(cazac))[0]
    t1 = time.time()
    cazac_time = time_string(t1-t0)
    print('Searching for CAZAC sequences...Done! '+cazac_time)
    np.savetxt(filename, cazacs)
    print('Phase values for CAZAC saved.')

    # Duplicate Checking
    t0 = time.time()
    print('Batch searching for duplicate CAZACs...', end='\r')
    exp_cazac = np.exp(2*np.pi*1j*cazacs)
    cazac_pool = [None]*pool_workers
    batch_size = int(simulations/pool_workers)
    for i in range(pool_workers):
        if i < pool_workers - 1:
            cazac_pool[i] = exp_cazac[batch_size*i:batch_size*(i+1),:]
        else:
            cazac_pool[i] = exp_cazac[batch_size*(i-1):,:]
    filter_cazac = lambda arr: cazac_filter(arr, tol=dup_tolerance, max_iter = batch_size)
    with Pool(pool_workers) as pool:
        unique_cazacs_pool = pool.map(filter_cazac, cazac_pool)
    t1 = time.time()
    batch_time = time_string(t1-t0)
    print('Batch filtering for duplicate CAZACs...Done! '+batch_time)
    unique_cazacs_pool = tuple([np.array(A) for A in unique_cazacs_pool])
    unique_cazacs_pool = np.vstack(unique_cazacs_pool)
    t0 = time.time()
    print('Filtering batches for duplicate CAZACs...',end='\r')
    unique_cazacs = cazac_filter(unique_cazacs_pool, tol=dup_tolerance, max_iter = unique_cazacs_pool.shape[0])
    t1 = time.time()
    filter_time = time_string(t1-t0)
    print('Filtering batches for duplicate CAZACs...Done! '+filter_time)
    print(f'Unique CAZACs found: {unique_cazacs.shape[0]}')
    np.savetxt(f'unique_cazac_length{n}_sim{simulations}_batch{batchnum}.csv', unique_cazacs)
    