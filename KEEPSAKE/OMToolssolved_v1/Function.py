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
    
    def get_constraint(self,k):
        z = np.column_stack((k,np.zeros(self.num_vertices)))
        z = self.projectxy0_to_Bspline(z)
        return z[:,2]

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
    
    def get_k_pts(self,contour):
        while len(contour) <= self.num_vertices:
            new_contour = np.empty((0,2))
            for i in range(len(contour)-1):
                midpt = np.array([np.mean(contour[i:i+2,0]),np.mean(contour[i:i+2,1])])
                new_contour = np.append(new_contour, np.array([contour[i,:]]), axis=0)
                new_contour = np.append(new_contour, np.array([midpt]),        axis=0)
            contour = np.append(new_contour, np.array([contour[i+1,:]]), axis=0)

        k = np.zeros((self.num_vertices,3))
        indeces = np.rint(np.linspace(0,contour.shape[0]-2,self.num_vertices))
        for i in range(self.num_vertices):
            k[i,0] = contour[int(indeces[i]),0]
            k[i,1] = contour[int(indeces[i]),1]
        return k
    
    def get_dist_vec(self,k):
        dist_vec = []
        for i in range(len(k)-1):
            dist_vec = np.append(dist_vec, np.sqrt((k[i,0]-k[i+1,0])**2 + (k[i,1]-k[i+1,1])**2))
        return dist_vec

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
        u_vec = np.ones(self.num_vertices)
        v_vec = np.ones(self.num_vertices)
        compute_surface_projection(
            self.order_uv, self.num_control_points_u,
            self.order_uv, self.num_control_points_v,
            self.num_vertices, max_iter,
            k.reshape(self.num_vertices * 3), 
            self.cps.reshape(self.num_control_points_u * self.num_control_points_v * 3),
            u_vec, v_vec,self.robustness
        )
        return u_vec, v_vec

    def projectxy0_to_Bspline(self,k):
        u_vec,v_vec = self.get_uv_from_k(k)
        basis = self.get_basis_from_uv(u_vec,v_vec,0,0)
        return basis.dot(self.cps)