import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import restoration
import abel.tools, abel.transform
import cv2         #OpenCV
#local import
import tools

__version__ = '0.0.5'


class VMI:
    
    def __init__(self, data: np.array, radious: int or list=921, center: list = [994, 1019], **kw):

        if data.dtype != np.float32:     #The standerd data type is float32
            self.data = data.astype(np.float32)
        else:
            self.data = data
        self.shape = data.shape

        if radious:
            kw['radious'] = radious
        else:
            self.radious = None
        if center:
            kw['center'] = center
        else:
            self.center = None
        self.update(**kw)
        

    def update(self, **kw):
        # center
        if 'center' in kw:
            self.cy, self.cx = kw['center']
        elif 'cx' in kw and 'cy' in kw:
            self.cy, self.cx = kw['cy'], kw['cx']
        # radious
        if 'radious' in kw:
            if type(kw['radious']) == int:
                self.ry, self.rx = kw['radious'], kw['radious']
            else:
                self.ry, self.rx = kw['radious']
        elif 'rx' in kw and 'ry' in kw:
            self.ry, self.rx = kw['ry'], kw['rx']


    def crop(self, **kw): # also place the image in the center
        """
        reduce the size of the 2D "array" by cutting the data in the edges 
        around the "center" so the output array will have a new (Square) shape
        equals to 2 * radious. 
        """
        if kw:
            self.update(**kw)
        assert self.cx, 'Set the Image Center value "cx = ??"'
        assert self.cy, 'Set the Image Center valus "cy = ??"'
        assert self.rx, 'Set the cropping radious "rx = ??"'
        assert self.ry, 'Set the cropping radious "ry = ??"'

        #Define edges
        xi = self.cx - self.rx
        xf = self.cx + self.rx
        yi = self.cy - self.ry
        yf = self.cy + self.ry

        self.data = self.data[yi:yf + 1, xi:xf + 1]  # x corespond to col
        self.shape = self.data.shape
        self.cx = self.rx
        self.cy = self.ry
    
     

    def truncate(self, **kw):
        # Update and Check
        if kw:
            self.update(**kw)
        assert self.cx, 'Set the Image Center value "cx = ??"'
        assert self.cy, 'Set the Image Center valus "cy = ??"'
        assert self.rx, 'Set the cropping radious "rx = ??"'
        assert self.ry, 'Set the cropping radious "ry = ??"'

        # creating a circle of ones
        mask = tools.creat_mask(size = self.shape, center = (self.cy, self.cx), radious = (self.ry, self.rx), place_true = 'inside', output_values = '1/0')       
        
        #apply the mask
        self.data *= mask                       


    def rotate(self, angle = 60, reshape = False):                    
        self.data = ndimage.rotate(self.data, angle, reshape = reshape)


    
    def deconv(self, psf, iter=5): 
        # Check for alternative: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html
        """"
        Richardson-Lucy deconvolution.
    
        Parameters
            image : ndarray
                Input degraded image (can be N dimensional).
            psf : ndarray
                The point spread function.
            iterations : int, optional
                Number of iterations. This parameter plays the role of regularisation.
            clip : boolean, optional
                True by default. If true, pixel value of the result above 1 or 
                under -1 are thresholded for skimage pipeline compatibility.
    
        Returns
            im_deconv : ndarray
        """

        self.data = restoration.richardson_lucy(
            image=self.data, psf=psf, iterations=iter, clip=False)
        
        
        
    def correct_image_distortion(self, method='lsq', radial_range=None, dr=0.5, dt=0.5,\
                                 smooth=0, ref_angle=None, inverse=False, return_correction=False) : # Does nothing Vancy!
        """
        For more information: Check this link(https://pyabel.readthedocs.io/en/latest/abel.html#image-processing-tools)
        """

        self.data = abel.tools.circularize.circularize_image(self.data, method=method, center=(self.cy, self.cx), \
                 radial_range = radial_range, dr=dr, dt=dt, smooth=smooth, ref_angle=ref_angle, \
                 inverse=inverse, return_correction=return_correction)
        
        if type(self.data) == tuple or type(self.data) == list:
            self.additional_outputs = self.data[1:]
            self.data = self.data[0] 
            
        
    def to_polar(self, origin=None, Jacobian=False, dr=1, dt=None):
        """
        abel.tools.polar.reproject_image_into_polar(data, origin=None, Jacobian=False, dr=1, dt=None)
        
        Reprojects a 2D numpy array (data) into a polar coordinate system. “origin” is a tuple of (x0, y0) 
        relative to the bottom-left image corner, and defaults to the center of the image.

        Parameters:
            data (2D np.array)
            origin (tuple) – The coordinate of the image center, relative to bottom-left
            Jacobian (boolean) – Include r intensity scaling in the coordinate transform. 
                This should be included to account for the changing pixel size that occurs during the transform.
            dr (float) – Radial coordinate spacing for the grid interpolation tests show that there is not much point in going below 0.5
            dt (float) – Angular coordinate spacing (in radians) if dt=None, dt will be set such that 
                the number of theta values is equal to the maximum value between the height or the width of the image.
       
        Returns:
            output (2D np.array) – The polar image (r, theta)
            r_grid (2D np.array) – meshgrid of radial coordinates
            theta_grid (2D np.array) – meshgrid of theta coordinates
            
        Notes
        source: (https://pyabel.readthedocs.io/en/latest/abel.html#image-processing-tools)
        """
        
        self.data, self.r_grid, self.theta_grid = abel.tools.polar.reproject_image_into_polar(self.data, origin=origin, Jacobian=Jacobian, dr=dr, dt=dt)
        
    
    def polar2cart(self,):
        """"
        x,y = abel.tools.polar.polar2cart(t.r_grid, t.theta_grid)
        
        NOT completed
        """
        pass
    
    
    
    def get_image_quadrants(self, reorient=True, symmetry_axis=None, use_quadrants=(True, True, True, True), symmetrize_method=u'average'):
        
        """
        source: (https://pyabel.readthedocs.io/en/latest/abel.html#image-processing-tools)
        
        Given an image (m,n) return its 4 quadrants Q0, Q1, Q2, Q3 as defined below.
        
        Parameters:
            IM (2D np.array) – Image data shape (rows, cols)
            
            reorient (boolean) – Reorient quadrants to match the orientation of Q0 (top-right)
            
            symmetry_axis (int or tuple) – can have values of None, 0, 1, or (0,1) and specifies 
                no symmetry, vertical symmetry axis, horizontal symmetry axis, and both vertical 
                and horizontal symmetry axes. Quadrants are added. See Note.
            
            use_quadrants (boolean tuple) – Include quadrant (Q0, Q1, Q2, Q3) in the symmetry combination(s) and final image
            
            symmetrize_method (str) – Method used for symmetrizing the image.
                average
                    Simply average the quadrants.
                fourier
                    Axial symmetry implies that the Fourier components of the 2-D projection should be real. Removing the imaginary components in reciprocal space leaves a symmetric projection.
            (ref: Overstreet, K., et al. “Multiple scattering and the density distribution of a Cs MOT.” Optics express 13.24 (2005): 9672-9682. http://dx.doi.org/10.1364/OPEX.13.009672)
        
        Returns:
            Q0, Q1, Q2, Q3 – shape: (rows//2+rows%2, cols//2+cols%2) all oriented in the same direction as Q0 if reorient=True
        Return type:
            tuple of 2D np.arrays
            
        Notes
            The symmetry_axis keyword averages quadrants like this:
             +--------+--------+
             | Q1   * | *   Q0 |
             |   *    |    *   |
             |  *     |     *  |               cQ1 | cQ0
             +--------o--------+ --(output) -> ----o----
             |  *     |     *  |               cQ2 | cQ3
             |   *    |    *   |
             | Q2  *  | *   Q3 |          cQi == combined quadrants
             +--------+--------+
            
            symmetry_axis = None - individual quadrants
            symmetry_axis = 0 (vertical) - average Q0+Q1, and Q2+Q3
            symmetry_axis = 1 (horizontal) - average Q1+Q2, and Q0+Q3
            symmetry_axis = (0, 1) (both) - combine and average all 4 quadrants
            
        source: (https://pyabel.readthedocs.io/en/latest/abel.html#image-processing-tools)
        """
    
        self.quadrants = abel.tools.symmetry.get_image_quadrants(self.data,\
                                            reorient=reorient, symmetry_axis=symmetry_axis, \
                                            use_quadrants=use_quadrants, \
                                            symmetrize_method=symmetrize_method)
        
        
        
    def put_image_quadrants(self, Q, symmetry_axis=None ):
        
        """
        source: (https://pyabel.readthedocs.io/en/latest/abel.html#image-processing-tools)
        Reassemble image from 4 quadrants Q = (Q0, Q1, Q2, Q3) The reverse process to get_image_quadrants(reorient=True)
        Note: the quadrants should all be oriented as Q0, the upper right quadrant
        
        Parameters:
            Q (tuple of np.array (Q0, Q1, Q2, Q3)) – Image quadrants all oriented as Q0 shape (rows//2+rows%2, cols//2+cols%2)
            +--------+--------+
            | Q1   * | *   Q0 |
            |   *    |    *   |
            |  *     |     *  |
            +--------o--------+
            |  *     |     *  |
            |   *    |    *   |
            | Q2  *  | *   Q3 |
            +--------+--------+
            
            original_image_shape (tuple) – (rows, cols)
                reverses the padding added by get_image_quadrants() for odd-axis sizes
                odd row trims 1 row from Q1, Q0
                odd column trims 1 column from Q1, Q2
                
            symmetry_axis (int or tuple) –
                impose image symmetry
                    symmetry_axis = 0 (vertical)   - Q0 == Q1 and Q3 == Q2
                    symmetry_axis = 1 (horizontal) - Q2 == Q1 and Q3 == Q0
                    
            Returns:
                IM –
                    Reassembled image of shape (rows, cols):
                    symmetry_axis =
                          None             0              1           (0,1)
                        
                         Q1 | Q0        Q1 | Q1        Q1 | Q0       Q1 | Q1
                        ----o----  or  ----o----  or  ----o----  or ----o----
                         Q2 | Q3        Q2 | Q2        Q1 | Q0       Q1 | Q1
                    
            Return type:
                np.array
        """
    
        self.data = abel.tools.symmetry.put_image_quadrants(Q=Q, \
                    original_image_shape = self.shape, \
                    symmetry_axis=symmetry_axis)
        
        
        
    def darkspot_filter(self, darkspot_r = 400, blur_size = (11,11), threshold = 0.85, gaussian_threshold = 0.25, rotation_angle = 3.5, iterations = 5 ):
        """
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_photo/py_inpainting/py_inpainting.html#inpainting
        https://scikit-image.org/docs/dev/auto_examples/filters/plot_inpaint.html
        """

        #STEP 1: Finding Dark Spot
        # Creat a mask (with radious = r) that limit the search for dark spot
        mask = tools.creat_mask(size = self.shape, center = (self.cy, self.cx), radious = darkspot_r, place_true = 'inside', output_values = '1/0')

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
        
    
    
    def show(self, colorbar=False, **kw):
        """
        To show the image
        in kw:
            vmin: min value in the gray scale
            vmax: max value in teh gray scale
            alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
        """

        plt.imshow(self.data, cmap='gray', **kw)
        if colorbar:
            plt.colorbar()
        #plt.show()
        
    
    def abel_transform(self,):
        
        self.data = abel.transform.Transform(self.data, method='basex', direction='inverse').transform
        
        
    def get_velocities(self, d_theta = 15, dp = 2):
        
        self.thetas = range(0, 46 , d_theta) # 45 + 1 because the rotation is semitric and we take two axis (x, y)
        if 45 in self.thetas: # at angel 45 degrees (st_h = st_v) therefore we want avoide the duplicate 
            shape = (self.rx + 1, len(self.thetas)*2 - 1)
        else:
            shape = (self.rx + 1, len(self.thetas)*2)
        self.lines = np.zeros(shape = shape, dtype = np.float32)      ##what if rx != ry ??
        for i,theta in enumerate(self.thetas):
            data_rot = ndimage.rotate(self.data, theta, reshape = False)
            
            st_h = data_rot[self.cy - int(dp/2): self.cy + 1 + int(dp/2), self.cx : ]
            st_v = data_rot[self.cy : , self.cx - int(dp/2): self.cx + 1 + int(dp/2)]
        
            h = np.sum(st_h, axis = 0)
            v = np.sum(st_v, axis = 1)
            
            self.lines[:,i] = h
            self.lines[:, -1 - i] = v
    
    
    def plot_velocities(self,):
        plots = plt.plot(self.lines)
        plt.legend(plots, range(0,91, self.thetas.step))
        plt.show()
            
        
        
        
        
        
    
    def __repr__(self):
        #self.show()
        return ''


"""
d = np.load('data.npy')
v = VMI(d)
"""

