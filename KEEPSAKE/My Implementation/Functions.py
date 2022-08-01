import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from skimage import measure

from lsdo_geo.splines.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.splines.surface_projection_py import compute_surface_projection

class Problem(object):

    def __init__(self,cps,order_uv,size_grid,num_control_points_u,num_control_points_v,num_points_uv,num_vertices,proj_robustness):
        self.cps = cps
        self.order_uv = order_uv
        self.size_grid = size_grid
        self.num_control_points_u = num_control_points_u
        self.num_control_points_v = num_control_points_v
        self.num_points_uv = num_points_uv
        self.num_vertices = num_vertices
        self.robustness = proj_robustness
        
        self.reg_term = 0
        self.reg_ind = 0
        self.res = size_grid/num_points_uv

    def compute_objective(self,x,lamb,C):
        # Least Squares Problem
        obj = np.sum(self.get_dist_vec(x)**2)
        # Lagrange Multipliers
        obj += np.sum(lamb*C)
        # Regularization
        obj += self.reg_term*np.sum(x[self.reg_ind,:]**2)
        return obj

    def compute_gradient(self,x,lamb,A):
        end = self.num_vertices-1
        dLdx = np.zeros(self.num_vertices)
        dLdy = np.zeros(self.num_vertices)
        # Least Squares Derivatives
        dLdx[0]   += 4*x[0,0] - 2*x[1,0] - 2*x[end,0]
        dLdy[0]   += 4*x[0,1] - 2*x[1,1] - 2*x[end,1]
        dLdx[end] += 4*x[end,0] - 2*x[end-1,0] - 2*x[0,0]
        dLdy[end] += 4*x[end,1] - 2*x[end-1,1] - 2*x[0,1]
        for i in np.arange(1,end):
            dLdx[i] += 4*x[i,0] - 2*x[i+1,0] - 2*x[i-1,0]
            dLdy[i] += 4*x[i,1] - 2*x[i+1,1] - 2*x[i-1,1]
        # Regularization
        dLdx[self.reg_ind] += self.reg_term*2*x[self.reg_ind,0]
        dLdy[self.reg_ind] += self.reg_term*2*x[self.reg_ind,1]
        # Combine
        grad = []
        for i in range(self.num_vertices):
            grad = np.append(grad,(dLdx[i],dLdy[i]))
        # Lagrange Multipliers
        grad += np.dot(A.T,lamb)
        return grad

    def compute_hessian(self,x,lamb):
        end = self.num_vertices-1
        # Least Squares Derivatives
        hess = np.diag(4*np.ones(2*self.num_vertices))
        hess += -2*np.diag(np.ones(2*self.num_vertices-2),2)
        hess += -2*np.diag(np.ones(2*self.num_vertices-2),-2)
        hess += -2*np.diag(np.ones(2),2*self.num_vertices-2)
        hess += -2*np.diag(np.ones(2),-2*self.num_vertices+2)
        # Lagrange Multipliers
        u_vec,v_vec = self.get_uv_from_k(x)
        basisx2 = self.get_basis_from_uv(u_vec,v_vec,2,0)
        basisxy = self.get_basis_from_uv(u_vec,v_vec,1,1)
        basisy2 = self.get_basis_from_uv(u_vec,v_vec,0,2)
        d2Ldx2  = lamb*basisx2.dot(self.cps)[:,2]
        d2Ldxdy = lamb*basisxy.dot(self.cps)[:,2]
        d2Ldy2  = lamb*basisy2.dot(self.cps)[:,2]
        for i in 2*np.arange(0,self.num_vertices):
            temp = int(i/2)
            hess[i,i]     += d2Ldx2[temp]
            hess[i+1,i+1] += d2Ldy2[temp]
            hess[i+1,i]   += d2Ldxdy[temp]
            hess[i,i+1]   += d2Ldxdy[temp]
        # Regularization
        hess[2*self.reg_ind,2*self.reg_ind] += 2*self.reg_term
        hess[2*self.reg_ind+1,2*self.reg_ind+1] += 2*self.reg_term
        return hess

    def compute_Amatrix(self,k):
        u_vec,v_vec = self.get_uv_from_k(k)
        basisx = self.get_basis_from_uv(u_vec,v_vec,1,0)
        basisxy = self.get_basis_from_uv(u_vec,v_vec,1,1)
        basisy = self.get_basis_from_uv(u_vec,v_vec,0,1)
        Ax = basisx.dot(self.cps)[:,2]
        Axy = basisxy.dot(self.cps)[:,2]
        Ay = basisy.dot(self.cps)[:,2]
        Amat = np.zeros((self.num_vertices,2*self.num_vertices))
        for i in np.arange(0,2*self.num_vertices,2,dtype=int):
            temp = int(i/2)
            Amat[temp,i]    = Ax[temp] + Axy[temp]
            Amat[temp,i+1]  = Ay[temp] + Axy[temp]
        return Amat

    def projectxy0_to_Bspline(self,k):
        u_vec,v_vec = self.get_uv_from_k(k)
        basis = self.get_basis_from_uv(u_vec,v_vec,0,0)
        return basis.dot(self.cps)
    
    def get_dist_vec(self,k):
        dist_vec = []
        for i in range(len(k)-1):
            dist_vec = np.append(dist_vec, np.sqrt((k[i,0]-k[i+1,0])**2 + (k[i,1]-k[i+1,1])**2))
        return dist_vec
    
    def get_k_pts(self,contour):
        while len(contour) <= self.num_vertices:
            new_contour = np.empty((0,2))
            for i in range(len(contour)-1):
                midpt = np.array([np.mean(contour[i:i+2,0]),np.mean(contour[i:i+2,1])])
                new_contour = np.append(new_contour, np.array([contour[i,:]]), axis=0)
                new_contour = np.append(new_contour, np.array([midpt]),        axis=0)
            contour = np.append(new_contour, np.array([contour[i+1,:]]), axis=0)

        k = np.zeros((self.num_vertices+1,3))
        indeces = np.rint(np.linspace(0,contour.shape[0]-1,self.num_vertices+1))
        for i in range(self.num_vertices+1):
            k[i,0] = contour[int(indeces[i]),0]
            k[i,1] = contour[int(indeces[i]),1]
        return k

    def get_basis_from_uv(self,u_vec,v_vec,du,dv):
        nnz = u_vec.size**2 * self.order_uv * self.order_uv
        data = np.zeros(nnz)
        row_indices = np.zeros(nnz, np.int32)
        col_indices = np.zeros(nnz, np.int32)

        get_basis_surface_matrix(
            self.order_uv, self.num_control_points_u, du, u_vec, 
            self.order_uv, self.num_control_points_v, dv, v_vec, 
            u_vec.size, data, row_indices, col_indices,
        )

        basis = sps.csc_matrix(
            (data, (row_indices, col_indices)), 
            shape=(u_vec.size, self.num_control_points_u * self.num_control_points_v),
        )
        return basis

    def get_uv_from_k(self,k):
        max_iter = 1000
        u_vec = np.ones(len(k))
        v_vec = np.ones(len(k))
        compute_surface_projection(
            self.order_uv, self.num_control_points_u,
            self.order_uv, self.num_control_points_v,
            len(k), max_iter,
            k.reshape(len(k) * 3), 
            self.cps.reshape(self.num_control_points_u * self.num_control_points_v * 3),
            u_vec, v_vec,self.robustness
        )
        return u_vec, v_vec

    def get_initial_lvset(self):
        ### Initial Vector
        u_vec = np.einsum('i,j->ij', np.linspace(0.,1.,self.num_points_uv), np.ones(self.num_points_uv)).flatten()
        v_vec = np.einsum('i,j->ij', np.ones(self.num_points_uv), np.linspace(0.,1.,self.num_points_uv)).flatten()
        basis = self.get_basis_from_uv(u_vec,v_vec,0,0)
        ### Evaluate Test Points & Find Contours
        pts = basis.dot(self.cps)
        phi = np.reshape(pts[:,2],(self.num_points_uv,self.num_points_uv))
        contours = measure.find_contours(phi, 0)
        if contours == []:
            raise BaseException('No contour detected!')
        if len(contours) > 1:
            raise BaseException('Multiple contours detected!')
        contour = contours[0]*self.res
        ### Get points along contour
        k = self.get_k_pts(contour)
        initial_pts = self.projectxy0_to_Bspline(k)
        return initial_pts