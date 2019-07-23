"""
VMI Core Analysis
"""
import numpy as np
import abel.tools, abel.transform    #PyAbel


__version__ = '0.1.0'


class Core:

    def basex_transform(self,):
        self.data = abel.transform.Transform(self.data, method='basex', direction='inverse',\
                                             center = self.center).transform


    def lin_basex_transform(self, proj_angles, legendre_orders = [0, 2]):
        return abel.transform.Transform(self.data, method="linbasex", center=self.center,
                                        transform_options=dict(basis_dir=None, return_Beta=True,
                                                               legendre_orders=legendre_orders,
                                                               proj_angles=proj_angles))
            
            
    def electron_count(self, angles:list=range(0, 91, 15), dt=2):
        """
        Takes a polar image and integrate it alonge the angle axis from angle = theta - dt 
        to angle = theta + dt. And return a line that starts from r = 0 to r = end. If there is 
        more than one angle, then each angle will have a line output.
        parameters:
            angles: list or array - integration angles
            dt: float or int - delta theta; width of the integration in degrees,
        output:
            lines: 2darray - shape = [size of the radious, number of lines]
            each line crosponds to one angle of angles
        """
        self.lines = np.zeros(shape = (self.polar_data.shape[0], len(angles)))
        for i,t in enumerate(angles):
            ti = (t - dt) * np.pi / 180  # To radians
            tf = (t + dt) * np.pi / 180  
            mask = (self.theta_grid > ti) & (self.theta_grid < tf)
            self.lines[:,i] = np.sum(np.where(mask, self.polar_data, 0), axis = 1)
   
      
        
    def calibrate_energy(self):
        """
        This Function Takes the electron count data (After the Abel Transform) and return calibrated 
        energy-axis using the seperation of rings in the abel transformed image, which we already know.
        
        Parametrs:
            electron_count: 2d_array 
            
        ...
            
        """
        pass
    
    
