import numpy as np
import seaborn as sns

from lsdo_viz.api import BaseViz, Frame

sns.set()



order_uv = 4
size_grid = 5
num_cpts_u = 5
num_cpts_v = 5
num_points_uv = 10
num_vertices = 120
proj_robustness = 20




class Viz(BaseViz):
    def setup(self):
        # self.use_latex_fonts()

        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=8.,
                width_in=12.,
                nrows=1,
                ncols=1,
                wspace=0.4,
                hspace=0.4,
            ), 1)

    def plot(self,
             data_dict_current,
             data_dict_all,
             limits_dict,
             ind,
             video=False):
        import Function
        import define
        cpts = define.get_cpts('circle',4,num_cpts_u, num_cpts_v,size_grid)
        Func = Function.MyProblem(cpts,order_uv,size_grid,num_cpts_u,num_cpts_v,num_points_uv,num_vertices,proj_robustness)
        k0 = Func.get_initial_contour()
        
        x = data_dict_current['x']
        y = data_dict_current['y']
        #c_all = data_dict_all['compliance_comp.compliance']
        self.get_frame(1).clear_all_axes()
        with self.get_frame(1)[0, 0] as ax:
            ax.plot(np.append(x,x[0]),np.append(y,y[0]),'-o')
            ax.plot(k0[:,0],k0[:,1],'-o')
            ax.axis([0,5,0,5])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        self.get_frame(1).write()
        
"""         with self.get_frame(1)[0, 1] as ax:
            sns.barplot(x=x, y=h, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        ['inputs_comp.h'],
                        fig_axis=0,
                        data_axis=0,
                        lower_margin=0.1,
                        upper_margin=0.1,
                        mode='broad',
                    ))
                ax.set_xlabel('x')
                ax.set_ylabel('h')

                ax.get_xaxis().set_ticks([])

        self.get_frame(1).write() """
