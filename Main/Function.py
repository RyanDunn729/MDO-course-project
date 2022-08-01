import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from skimage import measure

from lsdo_geo.splines.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.splines.surface_projection_py import compute_surface_projection

class MyProblem(object):

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

    def get_initial_uv(self):
        ### Initial Vector
        u_vec = np.einsum('i,j->ij', np.linspace(0.,1.,self.num_points_uv), np.ones(self.num_points_uv)).flatten()
        v_vec = np.einsum('i,j->ij', np.ones(self.num_points_uv), np.linspace(0.,1.,self.num_points_uv)).flatten()
        pts = self.get_vals_from_uv(u_vec,v_vec,0,0)
        ### Evaluate Test Points & Find Contours
        phi = np.reshape(pts[:,2],(self.num_points_uv,self.num_points_uv))
        contours = measure.find_contours(phi, 0)
        if contours == []:
            raise BaseException('No contour detected!')
        if len(contours) > 1:
            raise BaseException('Multiple contours detected!')
        contour = contours[0]*self.res
        # Increase Points along Existing Contour
        while len(contour) <= self.num_vertices:
            new_contour = np.empty((0,2))
            for i in range(len(contour)-1):
                midpt = np.array([np.mean(contour[i:i+2,0]),np.mean(contour[i:i+2,1])])
                new_contour = np.append(new_contour, np.array([contour[i,:]]), axis=0)
                new_contour = np.append(new_contour, np.array([midpt]),        axis=0)
            contour = np.append(new_contour, np.array([contour[i+1,:]]), axis=0)
        # Select evenly spaced points
        k = np.zeros((self.num_vertices,3))
        indeces = np.rint(np.linspace(0,contour.shape[0]-2,self.num_vertices))
        for i in range(self.num_vertices):
            k[i,0] = contour[int(indeces[i]),0]
            k[i,1] = contour[int(indeces[i]),1]
        # Get uv from contour projection
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

    def get_initial_contour(self):
        ### Initial Vector
        u_vec = np.einsum('i,j->ij', np.linspace(0.,1.,self.num_points_uv), np.ones(self.num_points_uv)).flatten()
        v_vec = np.einsum('i,j->ij', np.ones(self.num_points_uv), np.linspace(0.,1.,self.num_points_uv)).flatten()
        pts = self.get_vals_from_uv(u_vec,v_vec,0,0)
        ### Evaluate Test Points & Find Contours
        phi = np.reshape(pts[:,2],(self.num_points_uv,self.num_points_uv))
        contours = measure.find_contours(phi, 0)
        if contours == []:
            raise BaseException('No contour detected!')
        if len(contours) > 1:
            raise BaseException('Multiple contours detected!')
        contour = contours[0]*self.res
        # Select evenly spaced points
        k = np.column_stack((contour,np.zeros(len(contour))))
        # Get uv from contour projection
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
        pts0 = self.get_vals_from_uv(u_vec,v_vec,0,0)
        return pts0

    def get_vals_from_uv(self,u_vec,v_vec,du,dv):
        u_vec = np.array(u_vec)
        v_vec = np.array(v_vec)
        nnz = u_vec.size**2 * self.order_uv**2
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
        pts = basis.dot(self.cps)
        return pts