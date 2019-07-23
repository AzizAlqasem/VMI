"""
Created on Sun Jun 23 13:54:13 2019
"""
import numpy as np

__version__ = '0.1.0'


def creat_mask(size:list, center: list, radious: int or list , place_true:str = 'inside', \
               output_values:str = 'true/false', dr: int or float = None) -> np.ndarray: 
    """
    This function creat a mask of True/False or 1/0 in a circle shape
    
    parameters:
        size: list or tuple - the mask size (rows, colums)
        center: list or tuple - the circle center
        raidious: int or list - radious of the circle
        place_true: str - options:
            inside: creat a circle as a disk shape
            border: creat a circle that is False/0 inside and outside
            outside: creat a circle that is False/0 inside and at the border
        output_values: str - the mask is either True/False or 0/1
        dr: int or float - the width of the border (only applicable when "place_true = 'border'")
        
    output:
        2d-array of 0/1 or True/False
    
    """
    
    # Handel inputs
    ros, cos = size
    cy, cx   = center
    ry, rx = radious if type(radious) == list or type(radious) == tuple else (radious, radious)
    
    #Make a grid    
    y, x = np.ogrid[-cy:ros - cy, -cx : cos - cx]
    x = np.int64(x) # from int32 to int64
    y = np.int64(y)
    
    #creat the mask
    if place_true.lower() == 'inside':
        m = ry**2 * x**2 + rx**2 * y**2 <= rx**2 * ry**2
    elif place_true.lower() == 'border':
        if dr:
            m = (ry**2 * x**2 + rx**2 * y**2 <= rx**2 * ry**2) ^ ((ry+dr)**2 * x**2 +(rx+dr)**2 * y**2 <= (rx+dr)**2 * (ry+dr)**2)
        else:
            m = ry**2 * x**2 + rx**2 * y**2 == rx**2 * ry**2
    elif place_true.lower() == 'outside':
        m = ry**2 * x**2 + rx**2 * y**2 >= rx**2 * ry**2
    else:
        raise(NameError('place_true = {} is not an option! Please choose form [inside, outside, border]'.format(place_true)))

    #Output values
    if output_values.lower() == 'bool' or 't' in output_values.lower():
        return m
    
    elif '1' in output_values.lower():
        mask = np.zeros(shape = size, dtype = np.int8)
        mask[m] = 1
        return mask
    else:
        raise(NameError('output_valuse = {} is not an option! Please choose form ["bool" or "true/false" , 1/0]'.format(output_values)))
    
    