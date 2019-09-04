"""
Image Processing
"""
import numpy as np
from scipy import ndimage
from skimage import restoration      #scikit-image
import abel.tools, abel.transform    #PyAbel
import cv2                           #OpenCV
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#local import
import tools

__version__ = '0.1.1'


class Filter:
    
    def deconv(self, psf, iter=5): 
        # Check for alternative: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html
        """"
        Richardson-Lucy deconvolution.
    
        """
        self.data = restoration.richardson_lucy(
            image=self.data, psf=psf, iterations=iter, clip=False)
        
        
    def darkspot_filter(self, darkspot_r = 400, blur_size = (11,11), threshold = 0.85, gaussian_threshold = 0.25, rotation_angle = 3.5, iterations = 5 ):
        """
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_photo/py_inpainting/py_inpainting.html#inpainting
        https://scikit-image.org/docs/dev/auto_examples/filters/plot_inpaint.html
        """

        #STEP 1: Finding Dark Spot
        # Creat a mask (with radious = r) that limit the search for dark spot
        mask = tools.creat_mask(size = self.shape, center = self.center, radious = darkspot_r, place_true = 'inside', output_values = '1/0')

        #Apply the mask
        data_focus = self.data.copy() * mask
        
        for _ in range(iterations):
            #average the data
            data_focus_blu = cv2.blur(data_focus, blur_size)
            data_focus_blu *= mask     # The cv2.blur has a bug of leaving non zeros value outside data_focus circle .. so this to make sure that every thing outside the circle is zero
            
            #Take the ratio
            ratio = data_focus / data_focus_blu
            
            #Get ride of NAN and INF valus that have been generated from #/0
            ratio[np.isnan(ratio)] = 10     # We should be carful here! 10 is randomly choosen (as long as it is NOT lees than ONE)
            ratio[np.isinf(ratio)] = 10
            
            #filter the data based on a threshold value to get the dark spots
            dark_spots_mask = np.where(ratio <= threshold, 1, 0).astype(np.float32)
            
            # Blure the dark spots by gausian
            dark_spots_mask = cv2.GaussianBlur(dark_spots_mask, blur_size, 0)
            
            # Apply another threshold, to the darkspot
            dark_spots_mask = np.where(dark_spots_mask >= gaussian_threshold, True, False)
            
            #STEP 2: Correction
            #rotate data by theta
            data_focus_ccw = ndimage.rotate(data_focus, rotation_angle, reshape = False)
            data_focus_cw = ndimage.rotate(data_focus, -rotation_angle, reshape = False)
            
            #take the average
            data_focus_avg = (data_focus_ccw + data_focus_cw) / 2
            
            #Subtitute the average values around dark spots into the data
            data_focus[dark_spots_mask] = data_focus_avg[dark_spots_mask]    
        
        mask = np.where(mask == 1, True, False)   # convert the mask from 1/0 to true/false
        self.data[mask] = data_focus[mask]
        
    


class Tools:
    
    def crop(self): # also place the image in the center
        """
        reduce the size of the 2D "array" by cutting the data in the edges 
        around the "center" so the output array will have a new (Square) shape
        equals to 2 * radious. 
        """
        #Define edges
        cy, cx = self.center
        xi = cx - self.radious
        xf = cx + self.radious
        yi = cy - self.radious
        yf = cy + self.radious

        self.data = self.data[yi:yf + 1, xi:xf + 1]  # x corespond to col
        self.shape = self.data.shape
        self.center = (self.radious, self.radious)
    

    def truncate(self): #* Use np.where(condition, data, elese) for faster preformance
        # creating a circle of ones
        mask = tools.creat_mask(size = self.shape, center = self.center, \
                                radious = self.radious, place_true = 'inside',\
                                output_values = '1/0')               
        #apply the mask
        self.data *= mask                       


    def rotate(self, angle = 60, reshape = False):                    
        self.data = ndimage.rotate(self.data, angle, reshape = reshape)

    
    def get_image_quadrants(self, reorient=True, symmetry_axis=None, use_quadrants=(True, True, True, True), symmetrize_method=u'average'):
        self.quadrants = abel.tools.symmetry.get_image_quadrants(self.data,\
                                            reorient=reorient, symmetry_axis=symmetry_axis, \
                                            use_quadrants=use_quadrants, \
                                            symmetrize_method=symmetrize_method)
        
        
    
    def choose_max_value(self,):
        self.quadrant_max_values = np.maximum.reduce(self.quadrants) 
    
    
    def put_image_quadrants(self, Q, symmetry_axis=None):
        self.data = abel.tools.symmetry.put_image_quadrants(Q=Q, \
                    original_image_shape = self.shape, \
                    symmetry_axis=symmetry_axis)   


    def to_polar(self, Jacobian=True, dr=1, dt=np.pi/180):
        self.polar_data, self.r_grid, self.theta_grid = abel.tools.polar.reproject_image_into_polar(self.data, origin=self.center, Jacobian=Jacobian, dr=dr, dt=dt)

        
        
    def find_center(self, itr = 3, ywidth=300, xwidth = 300, width_decline = [0.5, 0.5], min_width = [20, 20], plot = True):
        """
        This function finds center by iterative gaussian fitting. 
        The algorithm does the following:
            1- Takes initional guessing - from the user - for the first gaussian fit. 
            2- Uses the new fitting parameters as initial guessing for the second fit.
            3- Reduces the fitting window around the center of the previous fit.
            4- repeats 2 & 3 for <itr> times.
        parametars:
            itr: int  - number of iteration of fitting.
            ywidth: int  - The initial fitting window for finding the center in the y-axis
            xwidth: int  - The initial fitting window for finding the center in the x-axis
            width_decline: list  - [y-axis, x-axis] the decline percentage of the fitting window.
            min_width: list  - [y-axis, x-axis] The minimum width that the fitting window can reach. 
            plot: bool  -  ploting the fitting
        """
       
        ps = np.zeros([2, self.shape[0]])
        for i in range(2):
            s = np.sum(self.data, axis = i)
            s = s/s.max()
            ps[i] = s
            
        def gause_fit(axis, mid_point, width):
            p0 = [1, mid_point, 10, 0.1]
            p_range = np.arange(mid_point - width , mid_point + width)
            p_data =  ps[axis][mid_point - width : mid_point + width]
            #fitting
            popt, _ = curve_fit(tools._gaussian, p_range, p_data, p0 = p0)
            pc = popt[1]            # the point at which the value is max.
            
            if plot:
                plt.plot(ps[axis], '-k', lw = 2)
                plt.plot(p_range, tools._gaussian(p_range, *popt), '--', lw = 2)
                plt.show()
            
            return pc 
        
        yc, xc = self.center
        while itr and (xwidth > min_width[1] or ywidth > min_width[0]):  
            xc = gause_fit(axis = 0, mid_point=int(xc), width=xwidth)
            yc = gause_fit(axis = 1, mid_point=int(yc), width=ywidth)
            
            xwidth -= int(width_decline[1]*xwidth)
            ywidth -= int(width_decline[0]*ywidth)
            itr -= 1
        return (yc, xc)
    
    
class Image(Tools, Filter):
    
    pass





class Images:
    
