"""
VMI Core Analysis
"""
import numpy as np
import abel.tools, abel.transform    #PyAbel
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#local
#import mytools.data as mtd   # This is from outside VMI: 'C:\\Users\\aziz_\\OneDrive\\Aziz\\python\\my moduls\\mytools\\data.py'

__version__ = '0.1.1'


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
   
      
        
    def calibrate_energy(self, dE = 2 , prominence = [0.005,1], peaks_range =[400,600], plot = False) :
        """
        This Function Takes the electron count data and return calibrated energy-axis using 
        the seperation between peaks.
        
        Parametrs:
            dE: int or flaot - The photon Energy (which is equivelent to the difference between two peaks).
            prominence: list [min, max] - "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html"
            peaks_range: [min, max] - 
            plot: bool - for debugging   
        
        ____________
        Note:
            Conversion_const = dE / peaks_difference_average
            Calibrated_energy_axis = conversion_const * range(line)**2    
        """
        
        sample_line = self.lines[:,0]/self.lines[:,0].max()
        self.energy_axis = np.arange(0, sample_line.size)**2
        i,f = peaks_range
        self.peaks, properties = find_peaks(sample_line[i:f], prominence = prominence)
        self.peaks = self.peaks + i
        energy_peaks = self.energy_axis[self.peaks]
        peaks_diff = np.average(np.diff(energy_peaks))
        conversion_const = dE / peaks_diff
        self.energy_axis = conversion_const * self.energy_axis
        
        if plot:
            plt.plot(self.energy_axis, sample_line)
            plt.plot(self.energy_axis[self.peaks], sample_line[self.peaks], 'x')
        


class Data:
    
    
    def normalize(data, n = 1):
        """
        normalize the data by dividing the data over maximum (n) values. If n > 1, all top values
        are going to be averaged, then, the data is divided by that average.
        """
        pass
        #top = mtd.top_values(arr = data, n = n)
        #return data / (np.sum(top) / np.len(top))
        

            