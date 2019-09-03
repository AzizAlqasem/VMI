"""
Utilities that support VMI
"""
__version__ = '0.1.0'


class Utility:
                
    # To have copy of the data - in case the user need to go one step back!
    def put_copy(self, copy, copy_name:str = 'copy'):
        self.copies[copy_name] = copy.copy()
    
    
    def get_copy(self, copy_name = 'copy'):
        return self.copies[copy_name].copy()
    
