import numpy as np
#local import
from core import Core
from image import Image
from present import Present
from utilities import Utility


__version__ = '0.1.1'


class VMI(Core, Image, Present, Utility):
    
    def __init__(self, data:np.array, center:list = None, radious:int = 921, **kw):
        
        #The standerd data type is float32
        if data.dtype != np.float32:     
            self.data = data.astype(np.float32)
        else:
            self.data = data
        
        self.shape = data.shape
        self.radious = radious
        
        if not center:
            self.center = [i//2 for i in self.shape]
            try:
                self.center = [int(round(i, 0)) for i in self.find_center(itr = 3, plot=False)]
            except:
               pass
        else:
            assert type(center) == list or type(center) == tuple
            self.center  = center
    
        # Save copy for the raw data
        self.copies = {}
        self.put_copy(copy_name='odata', copy=self.data)
        
      
   




"""
d = np.load('data.npy')
v = VMI(d)
"""

