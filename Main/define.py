import numpy as np
import Function

def get_cpts(name,upper,num_cpts_u, num_cpts_v,size_grid):
    cpts = np.zeros((num_cpts_u, num_cpts_v, 3))
    cpts[:, :, 0] = np.einsum('i,j->ij', np.linspace(0., size_grid, num_cpts_u), np.ones(num_cpts_v))
    cpts[:, :, 1] = np.einsum('i,j->ij', np.ones(num_cpts_u), np.linspace(0., size_grid, num_cpts_v))
    cpts[:, :, 2] = -0.5*np.ones((num_cpts_u, num_cpts_v))
    if name == 'I':
        cpts[4, 3:7, 2] = upper
        cpts[2:8, 2, 2] = upper
        cpts[2:8, 7, 2] = upper
    elif name == 'x':
        for i in np.arange(2,num_cpts_u-2):
            cpts[i,i,2] = upper
            cpts[i,num_cpts_u-1-i,2] = upper
    elif name == 'circle':
        cpts[2,2,2] = upper
    cpts = cpts.reshape((num_cpts_u * num_cpts_v, 3))
    return cpts