import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from skimage import measure
import Functions

# Initial Values
order_uv = 4
size_grid = 5
num_control_points_u = 10
num_control_points_v = 10
num_points_uv = 40
num_vertices = 15
proj_robustness = 20
sol_tol = 1e-6
res = size_grid/num_points_uv
# Control Points
cps = np.zeros((num_control_points_u, num_control_points_v, 3))
cps[:, :, 0] = np.einsum('i,j->ij', np.linspace(0., size_grid, num_control_points_u), np.ones(num_control_points_v))
cps[:, :, 1] = np.einsum('i,j->ij', np.ones(num_control_points_u), np.linspace(0., size_grid, num_control_points_v))
cps[:, :, 2] = -np.ones((num_control_points_u, num_control_points_v))
for i in np.arange(2,num_control_points_u-2):
    cps[i,i,2] = 2.5
    cps[i,num_control_points_u-1-i,2] = 2.5
cps = cps.reshape((num_control_points_u * num_control_points_v, 3))
#################################
def suff_dec(L,G,x,pk,lamb,C):
    alpha = 1
    test = prob.compute_objective(x+alpha*pk,lamb,C)
    while test > L + 1e-4*alpha*np.dot(G,pk.flatten()):
        alpha = alpha*0.99
        test = prob.compute_objective(x+alpha*pk,lamb,C)
        if alpha < 1e-10:
            return 1e-6
    return alpha
#################################
# Initialize Problem
prob = Functions.Problem(cps,order_uv,size_grid,num_control_points_u,num_control_points_v,num_points_uv,num_vertices,proj_robustness)
cpts_new = prob.get_initial_lvset()
#################################
u_vec = np.einsum('i,j->ij', np.linspace(0.,1.,num_points_uv), np.ones(num_points_uv)).flatten()
v_vec = np.einsum('i,j->ij', np.ones(num_points_uv), np.linspace(0.,1.,num_points_uv)).flatten()
basis = prob.get_basis_from_uv(u_vec,v_vec,0,0)
pts = basis.dot(cps)
phi = np.reshape(pts[:,2],(num_points_uv,num_points_uv))
contours = measure.find_contours(phi, 0)
contour = contours[0]*res
print(len(contour))
exit()
def check_pts(k):
    # 3d Plot
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(pts[:, 0], pts[:, 1], pts[:,2], 'k.')
    ax1.plot(cps[:, 0], cps[:, 1], cps[:, 2], 'or',markersize=1)
    ax1.plot(k[:, 0], k[:, 1], np.zeros(len(k)), 'b-', linewidth=3)
    plt.show()
    # 2d Contour Plot
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    ax2.plot(contour[:, 0], contour[:, 1], 'k-')
    ax2.plot(k[:, 0], k[:, 1], 'bo-', markersize=5, linewidth=1)
    ax2.axis([0, num_points_uv*res, 0, num_points_uv*res])
#################################
check_pts(contour)
exit()
# Lagrange Multipliers / Newton Update / Backwards Line Search
i = 0
lamb_new = np.ones(num_vertices)
while i < 350:
    # Update Values
    lamb = lamb_new
    cpts = cpts_new
    # Compute Values / Derivatives
    C = cpts[0:num_vertices,2]
    L = prob.compute_objective(cpts,lamb,C)
    A = prob.compute_Amatrix(cpts[0:num_vertices,:])
    dLdx = prob.compute_gradient(cpts,lamb,A)
    d2Ldx2 = prob.compute_hessian(cpts[0:num_vertices,:],lamb)
    # Solution Check
    if nplinalg.norm(dLdx)<sol_tol:
        print('Converged in ',i,' iterations')
        break
    # Solve KKT
    top = np.column_stack((d2Ldx2,A.T))
    bot = np.column_stack((A,np.zeros((num_vertices,num_vertices))))
    KKT = np.vstack((top,bot))
    b = np.append(dLdx,C)
    sol = nplinalg.solve(KKT,-b)
    pk = sol[0:2*num_vertices]
    dlamb = sol[2*num_vertices:3*num_vertices]
    # Line Search
    alpha = suff_dec(L,dLdx,cpts[0:num_vertices,0:2],pk.reshape((num_vertices,2)),lamb,C)
    # Update Next Step
    cpts_step = alpha*pk.reshape((num_vertices,2))
    cpts_new = cpts[:,0:2] + np.vstack((cpts_step,cpts_step[0,:]))
    cpts_new = np.column_stack((cpts_new,np.zeros((num_vertices+1,1))))
    cpts_new = prob.projectxy0_to_Bspline(cpts_new)
    lamb_new = lamb + alpha*dlamb
    i += 1
    check_pts(cpts)
    plt.pause(0.01)
    print('Constraint: ',nplinalg.norm(C))
    print('Objective: ',L)
    print('Grad Norm: ',nplinalg.norm(dLdx))
    print('alpha: ',alpha)
    print('Points')
    print(cpts)
print(prob.get_dist_vec(cpts))
print(np.var(prob.get_dist_vec(cpts)))
print('Constraint: ',nplinalg.norm(C))
print(cpts)
check_pts(cpts)
plt.show()