"""
#Visulization
Present VMI Data and Results 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


__version__ = '0.1.0'


# Setup figure defult values
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['figure.figsize'] = [8.0, 6.0]

class Present:
  
    def plot_electron_count_vs_energy(self, xi:int=None, xf:int=None, chose:list=None, yscale = 'log'):        
        chose = range(self.lines.shape[-1]) if chose == None else chose
        velocity_axis = np.arange(0,len(self.lines))[xi:xf]
        energy_axis = velocity_axis #** 2 / 2            # Not Calabrated yet
        
        plt.cla()
        self.plots = plt.plot(energy_axis,self.lines[xi:xf, chose])
        plt.xlabel('Energy (J)')
        plt.ylabel('Counts (a.u.)')
        plt.legend(self.plots, list(np.arange(90,-1,-15)[chose])) # The zero angle at the y-axis
        
    
    def plot_sep_electron_count(self, dpi = 200, figsize = [8,6], yscale = 'log'):
        fig, a = plt.subplots(4,2, dpi = dpi, figsize = figsize)
        a = a.flatten()
        ang = [0, 15, 30, 45, 60, 75, 90][::-1]
        for i,l in enumerate(self.lines.T):
            a[i].plot(l[10:500], label = ang[i])
            a[i].legend()
        plt.show()
    
    def show(self, colorbar=False, vmin = None, vmax = None):
        """
        To show the image
        in kw:
            vmin: min value in the gray scale
            vmax: max value in teh gray scale
            alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
        """
        plt.imshow(self.data, cmap='gray', vmin = vmin, vmax = vmax)
        if colorbar:
            plt.colorbar()
        #plt.show()
        

