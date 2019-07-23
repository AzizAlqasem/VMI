"""
VMI Code Notes
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import restoration      #scikit-image
import abel.tools, abel.transform    #PyAbel
import cv2                           #OpenCV
#local import
import tools



def plain_abel_trasformation():
    m = tools.creat_mask([201, 201], center = [100,100], radious=(70,70),output_values='1/0', place_true='inside')
    
    basex = abel.transform.Transform(m, method='basex', direction='inverse',\
                                             center = (100, 100)).transform
    linbasex = abel.transform.Transform(m, method='linbasex', direction='inverse',\
                                             center = (100, 100)).transform
    
    fig, a = plt.subplots(1,3, dpi = 200)
    a[0].imshow(m, cmap = 'gray')
    a[0].set_title('Data of 1/0')
    a[1].imshow(basex, cmap = 'gray', vmin = 0.001, vmax = 0.05)    
    a[1].set_title('Basex')
    a[2].imshow(linbasex, cmap = 'gray', vmin = 0.001, vmax = 0.05)
    a[2].set_title('LinBasex')
    plt.show()
