import numpy as np 
import cv2 
import os



def img_to_dataset(img, window_size=10, squeeze=True): 
    """
    Returns X, y for an input image.  Considers input window_size.  
      The dimensions + padding size are not considered in this function.   User beware.  

    Parameters: img (utils.Image)  
                window_size (int)
    
    Return:  X (np.array) of dim (N, window_size, window_size) or (N, window_size**2)     
             y (np.array) (BGR) of dim (N, 3)

    """

    X = list()
    y = list() 

    for i in range(100): 
        for j in range(100): 
            
            i_ = i + img.padding
            j_ = j + img.padding
            
            grays, b, g, r = img.get_dataset_for_pixel(i_, j_, window_size=window_size, squeeze=squeeze)
            X.append(grays)
            y.append((b, g, r))
            
            
    X = np.array(X)
    y = np.array(y)

    return X, y 


def reconstruct_from_vectors(blue, green, red, dimension=110):
    """ Reconstructs colored image form blue green and red channels. 
         Dimension arg is the dimension of the photograph.  Default is 110x100
            (5 for padding).  

        Use the following plt function to plot. 
        plt.imshow(cv2.cvtColor(reconstructed.astype('uint8'), cv2.COLOR_BGR2RGB))

    """ 

    blue_test = blue.reshape(-1, 1).squeeze()
    green_test = green.reshape(-1, 1).squeeze()
    red_test = red.reshape(-1, 1).squeeze()

    reconstructed = np.zeros(shape=(dimension, dimension, 3))

    reconstructed[:, :, 0] = blue_test.reshape(dimension, dimension)
    reconstructed[:, :, 1] = green_test.reshape(dimension, dimension)
    reconstructed[:, :, 2] = red_test.reshape(dimension, dimension)

    return reconstructed




def normalize(vector): 
    """ Normalize the pixel values 
    """
    return vector / 255 



def to_255_scale(vector): 
    """ Get 0-255 pixel value for an input vector.  Returns uint8 vector.  
    """
    return np.round(vector * 255).astype('uint8') 



def get_onehotencoding_vectors(indices_of_nearest_cluster, n_colors, y_list): 
    """ Given a 2D matrix of indecies, returns one-hot encoding vectors 
    """

    for i in range(indices_of_nearest_cluster.shape[0]):
        for j in range(indices_of_nearest_cluster.shape[1]): 
            
            idx = indices_of_nearest_cluster[i, j]
            one_hot = np.zeros(n_colors)
            one_hot[idx] = 1
            
            y_list.append(one_hot)
            
    return y_list


class Image():
    """ Holds a single image.  Contains helper functions for generating datasets
        from the image data.  Utilizes opencv2 for its image data utilities.  
    """


    def __init__(self, path, resize=100, padding=10, debug=False): 
        
        self.path = path 
        fullpath = os.path.join(os.getcwd(), path)
        self.data = cv2.imread(fullpath)
        if debug: print('Read in file {}.'.format(fullpath))
        self.original = self.data 
        self.data = cv2.resize(self.data, (resize, resize), interpolation = cv2.INTER_AREA)
        self.dim = resize 

        self.data_nopadding = self.data 
        self.gray_nopadding = self.convert_gray_nb()  

        self.padding = padding 
        self.add_padding(padding)

        self.get_BGR_channels()
        self.convert_gray() 

        return 
            
    def get_BGR_channels(self): 

        # .data is in BGR  
        self.blue_channel = self.data[:,:,0]
        self.green_channel = self.data[:,:,1]
        self.red_channel = self.data[:,:,2]
    
        return self.blue_channel, self.green_channel, self.red_channel  
    
    
    
    def convert_gray(self): 
        
        b = self.data[:, :, 0]
        g = self.data[:, :, 1]
        r = self.data[:, :, 2]
        
        self.gray = 0.21*r + 0.72*g + 0.07*b
        return self.gray
    

    def convert_gray_nb(self): 
        
        b = self.data_nopadding[:, :, 0]
        g = self.data_nopadding[:, :, 1]
        r = self.data_nopadding[:, :, 2]
        
        self.gray_nopadding = 0.21*r + 0.72*g + 0.07*b
        return self.gray_nopadding

    
    def add_padding(self, padding=10, color=0): 
        """  Adds padding to the image.  Modifies .data.   
        """
        self.data = cv2.copyMakeBorder(self.data, padding, padding, padding, 
                                       padding, cv2.BORDER_CONSTANT, value=color) 

        return self.data 

    





    
    def get_dataset_for_pixel(self, i, j, window_size=10, squeeze=True): 
        """ Returns gray values for a window around the target pixel at location 
            i, j.  The window_size will be rounded.  
            e.g window size of 11 --> 5 pixels to left, right, up and down of target 
        
            Of the window, gray pixel values are turned.  The target location's 
            rgb values are returned.  

            The blue, green, and red values represent the response feature, y.
            The gray array represents the feature array, X.  

            Note that the zeta = (window_size-1)/2 value SHOULD be less than or 
            equal to the padding of the image.  Be mindful of the padding when 
            choosing window size.   

            DO account for padding for i and j.  Padding is not accounted for in 
             i, j in this function.   
        """

        zeta = int((window_size-1)/2)

        # The BGR values represent the target features, y 

        b, g, r = self.data[i, j]

      
        # Gray represents the predictive features, X
        gr = self.gray[i - zeta  :  i + zeta + 1, 
                       j - zeta  :  j + zeta + 1]
      

        if squeeze: 
            gray = gr.reshape(-1, 1).squeeze() 
        else: 
            gray = gr 

        return gray, b, g, r 
