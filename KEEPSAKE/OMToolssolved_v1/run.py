import openmdao.api as om
import omtools.api as ot
import numpy as np
import Function
import matplotlib.pyplot as plt
#################################
# Initial Values
order_uv = 4
size_grid = 5
num_control_points_u = 5
num_control_points_v = 5
num_points_uv = 15
num_vertices = 10
proj_robustness = 20
reg_term = 1e-6
reg_index = 5
res = size_grid/num_points_uv
#################################
# Control Points
cps = np.zeros((num_control_points_u, num_control_points_v, 3))
cps[:, :, 0] = np.einsum('i,j->ij', np.linspace(0., size_grid, num_control_points_u), np.ones(num_control_points_v))
cps[:, :, 1] = np.einsum('i,j->ij', np.ones(num_control_points_u), np.linspace(0., size_grid, num_control_points_v))
cps[:, :, 2] = -np.ones((num_control_points_u, num_control_points_v))
cps[2, 2, 2] = 10
cps[2, 3, 2] = 1
cps[2, 1, 2] = 1
cps[1, 2, 2] = -5
cps[3, 2, 2] = -5
cps = cps.reshape((num_control_points_u * num_control_points_v, 3))
#################################
# Initialize Problem
Bspline = Function.Problem(cps,order_uv,size_grid,num_control_points_u,num_control_points_v,num_points_uv,num_vertices,proj_robustness)
x0 = Bspline.get_initial_lvset()[:,0:2]
w = x0[:,0]**2 + x0[:,1]**2
reg_index = int(np.dot((w == np.min(w)),np.arange(0,num_vertices)))
#################################
class MyProblem(ot.Group):
    def setup(self):
        # Inputs
        x = self.declare_input('cpts_x', val=x0[:,0])
        y = self.declare_input('cpts_y', val=x0[:,1])
        # Distance Variance Calculation
        end = num_vertices-1
        dist_sum = (x[end]-x[0])**2 + (y[end]-y[0])**2
        dist2_sum = ((x[end]-x[0])**2 + (y[end]-y[0])**2)**2
        x[1:] - x[:-1]
        for i in range(end):
            dist_sum = ot.sum(
                dist_sum,(x[i]-x[i+1])**2 + (y[i]-y[i+1])**2
            )
            dist2_sum = ot.sum(
                dist2_sum,((x[i]-x[i+1])**2 + (y[i]-y[i+1])**2)**2
            )
        objective = ot.sum(dist2_sum,-num_vertices*(dist_sum/num_vertices)**2)
        # Regularization
        reg = reg_term*(x[reg_index]**2 + y[reg_index]**2)
        # Constraint
        C = Bspline.get_constraint(x0)
        #C = Bspline.get_constraint(np.column_stack((x[:],y[:])))
        # Output of Comp
        self.register_output('Output',objective + reg)
        self.register_output('Const',C)
#################################
prob = om.Problem()
prob.model = MyProblem()
prob.model.add_design_var('cpts_x')
prob.model.add_design_var('cpts_y')
prob.model.add_objective('Output')
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['debug_print'] = ['desvars']
prob.driver.options['tol'] = 1.0e-10
prob.driver.options['maxiter'] = 250
prob.setup(force_alloc_complex=True)
prob.run_model()
print('Initial: ',np.column_stack((prob['cpts_x'],prob['cpts_y'])))
prob.run_driver()
#################################
print(np.column_stack((prob['cpts_x'],prob['cpts_y'])))

k = np.column_stack((prob['cpts_x'],prob['cpts_y']))
dist = Bspline.get_dist_vec(np.vstack((k,k[0,:])))
print(dist)
print(np.var(dist))

plt.plot(np.append(x0[:,0],x0[0,0]),np.append(x0[:,1],x0[0,1]),label='Initial')
plt.plot(np.append(prob['cpts_x'],prob['cpts_x'][0]),np.append(prob['cpts_y'],prob['cpts_y'][0]),label='Final')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right', shadow=True, fontsize='medium')
plt.axis([0, 5, 0, 5])

plt.show()