import openmdao.api as om
import omtools.api as ot
import numpy as np
from Function import MyProblem 
from define import get_cpts 
import matplotlib.pyplot as plt

#from lsdo_viz.api import Problem
#################################
# Initial Values
order_uv = 4
size_grid = 5
num_cpts_u = 5
num_cpts_v = 5
num_points_uv = 10
num_vertices = 40
proj_robustness = 20
#################################
# Shape Selection
cpts = get_cpts('circle',4,num_cpts_u, num_cpts_v,size_grid)
#################################
# Initialize Problem
Func = MyProblem(cpts,order_uv,size_grid,num_cpts_u,num_cpts_v,num_points_uv,num_vertices,proj_robustness)
u0,v0 = Func.get_initial_uv()
#################################
class uv2xyz(om.ExplicitComponent):
    def setup(self):
        self.add_input('u',shape=num_vertices)
        self.add_input('v',shape=num_vertices)
        self.add_output('x',shape=num_vertices)
        self.add_output('y',shape=num_vertices)
        self.add_output('z',shape=num_vertices)
        self.declare_partials(of='*', wrt='*')
    
    def compute(self, inputs, outputs):
        outputs['x'] = Func.get_vals_from_uv(inputs['u'],inputs['v'],0,0)[:,0]
        outputs['y'] = Func.get_vals_from_uv(inputs['u'],inputs['v'],0,0)[:,1]
        outputs['z'] = Func.get_vals_from_uv(inputs['u'],inputs['v'],0,0)[:,2]
    
    def compute_partials(self, inputs, partials):
        du = Func.get_vals_from_uv(inputs['u'],inputs['v'],1,0)
        dv = Func.get_vals_from_uv(inputs['u'],inputs['v'],0,1)
        partials['x','u'] = np.diag(du[:,0])
        partials['x','v'] = np.diag(dv[:,0])
        partials['y','u'] = np.diag(du[:,1])
        partials['y','v'] = np.diag(dv[:,1])
        partials['z','u'] = np.diag(du[:,2])
        partials['z','v'] = np.diag(dv[:,2])
#################################
class DistVariance(ot.Group):
    def setup(self):
        self.create_indep_var('u',shape=num_vertices)
        self.create_indep_var('v',shape=num_vertices)
        self.add_subsystem('Bspline',uv2xyz(),promotes=['*'])
        
        x = self.declare_input('x',shape=num_vertices)
        y = self.declare_input('y',shape=num_vertices)
        # Distance Variance Calculation
        end = num_vertices-1
        dist_sum = (x[end]-x[0])**2 + (y[end]-y[0])**2
        dist2_sum = ((x[end]-x[0])**2 + (y[end]-y[0])**2)**2
        for i in range(end):
            dist_sum = ot.sum(
                dist_sum,(x[i]-x[i+1])**2 + (y[i]-y[i+1])**2
            )
            dist2_sum = ot.sum(
                dist2_sum,((x[i]-x[i+1])**2 + (y[i]-y[i+1])**2)**2
            )
        objective = ot.sum(dist2_sum,-num_vertices*(dist_sum/num_vertices)**2)
        # Output of Comp
        self.register_output('Output',objective)
#################################
prob = om.Problem()
prob.model = DistVariance()
prob.model.add_design_var('u',lower=0,upper=1)
prob.model.add_design_var('v',lower=0,upper=1)
prob.model.add_objective('Output')
prob.model.add_constraint('z',equals=0)
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
#prob.driver.opt_settings['Major feasibility tolerance'] = 1e-10
#prob.driver.options['tol'] = 1e-10
#prob.driver.options['maxiter'] = 250
prob.setup(force_alloc_complex=True)
prob['u'] = np.array(u0)
prob['v'] = np.array(v0)

#prob.run()
prob.run_model()
prob.run_driver()
#################################
""" pts = Func.get_vals_from_uv(np.array(prob['u']),np.array(prob['v']),0,0)
k0 = Func.get_initial_contour()

plt.plot(np.append(pts[:,0],pts[0,0]),np.append(pts[:,1],pts[0,1]),'-o',label='Optimized Contour')
plt.plot(k0[:,0],k0[:,1],'-o',label='Initial Contour')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right', shadow=True, fontsize='medium')
plt.axis('equal')
plt.axis([0,5,0,5])
plt.show() """