import numpy as np
#local import
from core import Core
from image import Image
from present import Present
from utilities import Utilities


__version__ = '0.1.0'


class VMI(Core, Image, Present, Utilities):
    
    def __init__(self, data:np.array, radious:int = 921, center:list=[994, 1019], **kw):
        
        #The standerd data type is float32
        if data.dtype != np.float32:     
            self.data = data.astype(np.float32)
        else:
            self.data = data
        
        self.shape = data.shape
        self.radious = radious
        self.center  = center
    
        # Save copy for the raw data
        self.copies = {}
        self.put_copy(copy_name='odata', copy=self.data)
        
      
   




"""
d = np.load('data.npy')
v = VMI(d)
"""

