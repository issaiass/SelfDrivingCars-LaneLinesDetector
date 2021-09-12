# Do relevant imports

import matplotlib.pyplot as plt # for graphing purposes
import numpy as np              # for array operations
import cv2                      # for the image processing pipeline
import os                       # for path arranging despite the os

#  Class for Road Lines

class LaneLinesDetector:
    """ A Road Line Detector Class for detecting lane lines by pure image processing
    
    Input:
    None
       
    Output:
    A marked image with the road lines.  Not a perfect algorithm but still functional ;).
    """
    
    def __init__(self):
        """ Class initializer
    
        Input:
        None
       
        Output:
        None, just pass for the class to be initialized.
        """
        pass

    def canny(self, img, ksize=5, low_thr=50, high_thr=150):
        """ An edge detector pipeline based on canny edge and blurring.
    
        Input:
        img. np.array().  The image array of a single or multiple channel.
        ksize. int.  The kernel size, the number must be odd.
        low_thr. int.  The threshold composed of the low level of sensitivity for the canny edge detector.
        hight_thr. int.  The threhold composed of the highest level of sensitivity for the canny edge detector
       
        Output:
        img. np.array().  The array composed of the image in a single channel, borders only.
        """        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # convert to gray
        blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)  # blur some ammount to smooth borders
        canny = cv2.Canny(blur, low_thr, high_thr)        # edge detection
        return canny                                      # return the boundaries detected

    def get_roi(self, img, vertices, color=(255, 255, 255)):
        """ A simple part of the pipeline that crops an irregular regiond by its vertices.
    
        Input:
        img. np.array().  The image array of a single or multiple channel.
        vertices. np.array().  With format [(x0,y0), (x1, y1).., (xn, yn)] is the array of points to crop.
        color. tuple.  color of the region to mask, generally a white mask to "AND"it
       
        Output:
        mask. np.array().  The array composed of the image masked in a single channel array.
        """            
        mask = np.zeros_like(img)                          # create a white canvas
        cv2.fillPoly(mask, pts = [vertices], color=color)  # fill the canvas with the part to be anded
        mask = cv2.bitwise_and(img, mask)                  # make the mask
        return mask                                        # retur the mask
    
    def compute_hough(self, img, houghparams = dict([("rho" , 4), ("theta" , np.pi/180), ("threshold" , 15), 
                                                     ('min_line_lenght', 40), ('max_line_gap', 19)]),
                      color=(255,0,0), thickness=5):
        """ A simple part of the pipeline that crops an irregular regiond by its vertices.
    
        Input:
        img. np.array().  The image array of a single or multiple channel.
        vertices. np.array().  With format [(x0,y0), (x1, y1).., (xn, yn)] is the array of points to crop.
        houghparams. dict().  A dictionary composed of different houhg line detector parameters that consists of...
            rho. int. the distance resolution in pixels of the hough grid.
            theta. int. the angular resolution in radians of the hough grid.
            threshold.  int.  the minimum number of votes or intersection in hough grid cell.
            min_line_lenght. int. the minimun number of pixels making up a line.
            max_line_gap. int. maxixmum gain in pixels between connectable line segments.
        color. tuple. A tuple that consists of the color of the lines.
        thickness. int. The widht of the road lines marks.
       
        Output:
        lines. np.array().  A union of points in lines detected.
        """             
        rho = houghparams['rho']                          # resolution in pixels
        theta = houghparams['theta']                      # angular resolution in radians
        threshold = houghparams['threshold']              # minimum number of intersections
        min_line_length = houghparams['min_line_lenght']  # minimum number of pixels making up a line
        max_line_gap = houghparams['max_line_gap']        # maximum gap in pixels between connectable line segments
        
        # Run the probabilistic hough detector
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        return lines                                      # return the lines detected

    def draw_lines_simple(self, img, lines, color=(255,0,0), thickness=5):
        """ Draw the lines based on simple tuple extraction.
    
        Input:
        img. np.array().  The image array of a single or multiple channel.
        lines. np.array().  A union of points in lines detected.
        color. tuple. A tuple that consists of the color of the lines.
        thickness. int. The widht of the road lines marks.
       
        Output:
        img. np.array().  The image array of a 3 channel color image.
        """        
        if np.sum(lines) == None:                         # for error handling, if no lines, return the image
            return img
        
        img = np.zeros_like(img)                          # start by making a canvas
        if len(img.shape) == 2:                           # if the image is of 1ch
            img = cv2.merge((img,img,img))                # make a 3ch image
        for line in lines:                                # for every line
            x1,y1,x2,y2 = line.reshape(4)                 # reshape the points
            cv2.line(img, (x1, y1), (x2, y2), color, thickness) # graph over the image a line
        return img                                        # return the marked image
        
    def draw_lines_average(self, img, lines, color=(255,0,0), thickness=5):
        """ Draw the lines based on simple tuple extraction.
    
        Input:
        img. np.array().  The image array of a single or multiple channel.
        lines. np.array().  A union of points in lines detected.
        color. tuple. A tuple that consists of the color of the lines.
        thickness. int. The widht of the road lines marks.
       
        Output:
        img. np.array().  The image array of a 3 channel color image.
        """
        img = np.zeros_like(img)                          # make a canvas
        if len(img.shape) == 2:                           # if is a 1ch image
            img = cv2.merge((img,img,img))                # make it a 3ch image
        lt_line, rt_line = list(), list()                 # create a list of lines
        for line in lines:                                # for each line
            x1,y1,x2,y2 = line.reshape(4)                 # get the list of points
            params = np.polyfit((x1,x2), (y1,y2), 1)      # try to make a polygon fit by coordinates
            slope, intercept = params[0], params[1]       # get the parameters of a straight line
            if slope < 0:                                 # if is the left line (slope is negative)
                lt_line.append((slope, intercept))        # append to the left line list
            else:                                         # on the other hand
                rt_line.append((slope, intercept))        # append to the right line list
        lt_params = np.average(lt_line, axis=0)           # average each parameter listed
        rt_params = np.average(rt_line, axis=0)           # average each parameter listed
        
        # exception handling and bypassing section
        try:                                              # get the slopes of left and right lines
            slope1 = lt_params[0]                        
            slope2 = rt_params[0]
        except:
            return img                                    # return the image if an exception occurs
        if abs(slope1) < 0.2 or abs(slope2) < 0.2:        # if is not a good slope return the image
            return img
        # end of exception handling and bypassing section
        
        h = img.shape[0]                                  # get the height
        lt_coords = self.get_coords(h, lt_params)         # get the left coordinates
        rt_coords = self.get_coords(h, rt_params)         # get the right coordinates
        
        p1l, p1r = lt_coords                              # get the left points
        p2l, p2r = rt_coords                              # get the right points
        cv2.line(img, p1l, p1r, color, thickness)         # draw the left line
        cv2.line(img, p2l, p2r, color, thickness)         # draw the right line
        return img                                        # return the image

    def get_coords(self, h, params):
        """ get the coordinates based on line parameters.
    
        Input:
        h. int. The height of the image (max height).
        params. tuple.  The slope and intercept parametes of a line
       
        Output:
        (x1, y1), (x2, y2). tuple.  The tuple containing the points of the corresponding line.
        """        
        slope, intercept = params                         # get slope and intercept parameters
        y1 = h                                            # get the left line max height
        y2 = int((19/32)*h)                               # get the other point by a croped height
        x1 = int((y1 - intercept)/slope)                  # calculate the left position of the line
        x2 = int((y2 - intercept)/slope)                  # calculate the right position of the line
        return (x1, y1), (x2, y2)                         # return the point coordinates

    # lines processing pipeline
    def process(self, img, params, vertices, avg=True):
        """ Complete image process pipeline.
    
        Input:
        img. np.array().  The image as an array
        params.  dict().  the canny edge and hough parametes, color and threshold.  
                          See canny and compute_hough functions.
        vertices. np.array().  An array of the vertices.
        avg.  bool.  A boolean value for the image pipeline to process.
       
        Output:
        img.  np.array(). The final image of the pipeline
        """           
        
        # dictionary parameters, see respective functions canny, get_roi, compute_hough, get_lines_* for each parameters
        ksize = params['ksize']
        low_thr = params['low_thr']
        high_thr = params['high_thr']
        rho = params['rho']
        theta = params['theta']
        threshold=params['threshold']
        min_line_lenght=params['min_line_lenght']
        max_line_gap=params['max_line_gap']
        color=params['color']
        thickness=params['thickness']
    
        # complete hough dictionary
        houghparams=dict({'rho':rho, 'theta':theta, 'threshold':threshold,
                          'min_line_lenght':min_line_lenght, 'max_line_gap':max_line_gap
                         })
        
        # Do a canny edge detection
        canny = self.canny(img, ksize=ksize, low_thr=low_thr, high_thr=high_thr)
        # mark the roi
        roi = self.get_roi(canny, vertices)
        # compute lines by probabilistic hough
        lines = self.compute_hough(roi, houghparams=houghparams) 
        # process line average or simple lines
        if avg:
            marked_lines = self.draw_lines_average(roi, lines, color=color, thickness=thickness) # by average
        else: 
            marked_lines = self.draw_lines_simple(roi, lines, color=color, thickness=thickness)  # by simple marking
        img = cv2.addWeighted(img, 0.8, marked_lines, 1, 0)           # compute weighted images
        return img                                                    # return the image


if __name__ == '__main__':
    imgpath = os.path.join('test_images', 'solidYellowLeft.jpg')      # load the path of the image
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)                       # load the image

    # get initial parameters of vertices and the key & vals pairs
    ysize=img.shape[0]                                                # size of the image
    horizon = 300                                                     # region to crop
    vertices = np.array([[100, ysize], [500, horizon], [900, ysize]]) # the vertices of the region of interest

    # key value pairs for the algorithms used
    key_vals_pair = {'ksize': 5, 'low_thr': 50, 'high_thr': 150, # canny edge detector parameters
                     'rho': 4, 'theta': np.pi/180, 'threshold': 15, 'min_line_lenght': 40, 'max_line_gap': 19, # hough lines parameters
                     'color':(0, 0, 255), 'thickness': 20}            # drawing paramters

    params = dict(key_vals_pair)                                      # match the key and values

    ld = LaneLinesDetector()                                          # process the road lanes
    frame = ld.process(img, params, vertices, avg=True)               # load, convert to gray, blur, do canny, get the roi, draw lines
    cv2.imshow('RoadLines', frame)
    cv2.waitKey(0);                                                   # get the image 
    