# -*- coding: utf-8 -*-
"""
Functions to parse and process images using python-bioformats.

License: GPLv3

Author: Josh Lawrimore
"""

def start_vm(max_heap_size='6G'):
    """
    Start the Java Virtual Machine session.

    max_heap_size is a string that specifies how much memory the virual machine
    allowd to access. Default is 6 Gigabytes
    """
    import javabridge as jv
    import bioformats as bf
    #Start the Java virtual machine
    jv.start_vm(class_path=bf.JARS, max_heap_size=max_heap_size)

def kill_vm():
    """
    End the Java Virtual Machine session.
    """
    import javabridge as jv
    jv.kill_vm()



def im_compile(filename, path=None, rescale=False):
    """
    Converts image stack to a numpy nd array using python-bioformats.

    Parses the metadata of an image file and constructs a numpy ndarray of the
    images based on the number of channels, z-planes, and timepoints.

    Args:
        filename: Filename of the image to parse.
        path: The path to the image file.

    Returns:
        An ndarray of the images in the order YXCZT.
    """
    import bioformats as bf
    import numpy as np
    import os

    if path is None:
        fullfile = filename
    elif isinstance(path, str):
        fullfile = os.path.join(path, filename)
    metadata = bf.get_omexml_metadata(fullfile)
    ome_data = bf.OMEXML(xml=metadata)
    mat = np.zeros([ome_data.image().Pixels.SizeY,
                    ome_data.image().Pixels.SizeX,
                    ome_data.image().Pixels.channel_count,
                    ome_data.image().Pixels.SizeZ,
                    ome_data.image().Pixels.SizeT])
    with bf.ImageReader(fullfile) as rdr:
        for c_cnt in range(ome_data.image().Pixels.channel_count):
            for t_cnt in range(ome_data.image().Pixels.SizeT):
                for z_cnt in range(ome_data.image().Pixels.SizeZ):
                    image = rdr.read(c=c_cnt, t=t_cnt, z=z_cnt,
                                     rescale=rescale)
                    mat[:, :, c_cnt, z_cnt, t_cnt] = image
    return mat.astype(ome_data.image().Pixels.PixelType)

def max_int_proj(image_array, z_axis, squeeze=True):
    """
    Generate a maximum intensity projection of an ndarray image stack

    Uses numpy sum to sum all the images of the z-dimension to generate a
    maximum intensity projection and squeeze the resulting ndarray.

    Args:
        image_array: An ndarray containing images.
        axis: The axis about which to perform the sum operation
        squeeze: Perform squeeze on resutling ndarray?

    Returns:
        An ndarray of the maximum intensity projection.
    """
    if squeeze:
        return image_array.sum(axis=z_axis).squeeze()
    return image_array.sum(axis=z_axis)

def deint_zt(image_array, num_channels, num_zsteps, num_timepoints):
    """
    Deinterleave the z and t channels into two separate dimensions.

    Reshapes the ndarray of the image hyperstack by to a specified set of
    dimensions. Order of returned ndarray is [Y, X, Channels, Z, Timepoints].
    Since the order of the interleaved image planes is z1t1, z2t1, z1t2, z2t2
    z1t3, z12t3, the ndarray must first be reshaped by the number of timepoints
    and then the number of zsteps. However, this creates an improper order, so
    we must swap the last two axes.

    Args:
        image_array: An ndarray containing images with the dimension order
            [Y, X, C, Z, T]
        axis_to_split: The axis that contains two or more dimensions we want to
            split.
        channels: Number of channels in the hyperstack.
        zsteps: Number of z-steps per timepoints in the hyperstack.
        timepoints: Number timepoints in the hyperstack.

    Returns:
        An ndarray of images.
    """

    image_array = image_array.reshape([image_array.shape[0],
                                       image_array.shape[1],
                                       num_channels,
                                       num_timepoints,  #actually z-dimension
                                       num_zsteps])  #actually t-dimension
    image_array = image_array.swapaxes(-2, -1)
    return image_array

def scale_8bit(image_array):
    """
    Scale image intensity values in a hyperstack to 8-bit values and convert to
    type to uint8.

    Subtracts the minimum value from each element of the ndarray to set the
    minimum to 0. Then divides each element of the ndarray by the maximum value
    of the array to set the max intensity value to 1. Then multiplies each
    value by 255 and changes type to uint8.

    Args:
        image_array: An ndarray containing images with the dimension order
            [Y, X, C, Z, T]

    Returns:
        An ndarray of images with an uint8 data type.
    """

    image_array = image_array - image_array.min()
    return (image_array/image_array.max()*255).astype('uint8')

def blur_otsu_contours(image, ksize=9):
    """
    Returns contours after median blurring and generating a binary image by
    applying an Otsu-based threshold on a single 8bit image.

    Apply cv2.medianBlur on an image, then apply cv2.threshold using
    cv2.THRESH_BINARY+cv2.THRESH_OTSU. Then use cv2.findContours with
    cv2.RETR_EXTERNAL and cv2.CHAIN_APPROX_SIMPLE.

    Args:
        image: A grayscale, 8bit image
        ksize: Kernel size of the medianBlur filter

    Returns:
        A list of contours packaged as numpy arrays
    """

    import cv2

    blur = cv2.medianBlur(image, ksize)
    _, otsu = cv2.threshold(blur, blur.min(), blur.max(),
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_bounds(contours):
    """
    Converts contours to a list of numpy arrays containing the minimum x and y
    coordinates (the top-left corner) and the width and height of a rectangle
    fitting the contours.

    Iterate over all contours and use cv2.boundingRect to calculate the x and y
    coordinates and the with and height of a bounding rectangle.

    Args:
        contours: A list of numpy arrays containing contours

    Returns:
        A list of lists containing [x,y,w,h] where x and y are the minimum
        values (top-left corner of image), w is the width, and h is the height.
    """

    import cv2

    bounds = []
    for contour in contours:
        x_coord, y_coord, width, height = cv2.boundingRect(contour)
        bounds.append([x_coord, y_coord, width, height])
    return bounds

def crop_images(image_array, bounds, square_override=True, square_side=45):
    """
    Generate a list of cropped images.

    Crops an image array in the x and y axis using bounds to determine top-left
    corner position. By default, cropped image is a square with a with of 45
    pixels. Can use width and heigth in bounds array if crop_sqare is False.

    Args:
        image_array: An ndarray containing images with the dimension order
            [Y, X, C, Z, T]
        bounds: A list of lists containing [x,y,w,h] where x and y are the
            minimum values (top-left corner of image), w is the width, and h is
            the height.
        square_override: Toggles that cropped images should be square with a
            length specified by square_side. If False, width and height are set
            by the bounds
        square_side: The size, in pixels, of the side of the cropped image.
    """
    im_list = []
    for bound in bounds:
        if square_override:
            im_list.append(
                image_array[bound[1]:bound[1]+square_side,
                            bound[0]:bound[0]+square_side])
        else:
            im_list.append(
                image_array[bound[1]:bound[1]+bound[3],
                            bound[0]:bound[0]+bound[2]])
    return im_list

def write_hyper(image_array, filename, path=None):
    """
    Writes an image array as a multiplane TIFF file.

    Iterates over the dimensions of the image array and writes the TIFF image
    file using python bioformats write_image function.

    Args:
        image_array: An ndarray containing images with the dimension order
            [Y, X, C, Z, T]
        filename: The name of the TIFF file you want to save.
        path: The directory you want to save the TIFF file to. None will save
        to current working directory
    """
    import bioformats as bf
    import os

    if path:
        fullfile = os.path.join(path, filename)
    else:
        fullfile = filename
    if image_array.dtype == 'uint16':
        pixel_type = bf.PT_UINT16
    elif image_array.dtype == 'uint8':
        pixel_type = bf.PT_UINT8
    else:
        pixel_type = bf.PT_UINT16
    [_, _, c_size, z_size, t_size] = image_array.shape
    #The order the hyperstack is written is critical
    #This will write the hypderstack in XYCTZ order
    for t_idx in range(t_size):
        for z_idx in range(z_size):
            for c_idx in range(c_size):
                bf.write_image(fullfile,
                               image_array[:, :, c_idx, z_idx, t_idx],
                               pixel_type,
                               c=c_idx, z=z_idx, t=t_idx,
                               size_c=c_size, size_z=z_size, size_t=t_size)

def histomaker (image,im_dir, bin_num):
    """
    Create a histogram of pixel intensities for a single image
    
    Uses pillows to open the selected .tif image. Converts pixel intensities into 
    a 2D numpy array. The 2D numpy array is then converted to a 1D numpy array, 
    which is then plotted as a histogram. 
    
    Args:
        image: The image file name
        im_dir: The directory in which the image file exists
        bin_num: The number of bins needed for the histogram
        
    Returns: 
        A histogram with a user determined number of bins for a single image.

    """
    import os
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    
    os.chdir(im_dir)
    
    im = Image.open(image)
    imarray = np.array(im)
    imarray_1d = np.ravel(imarray)

    histogram = plt.hist(imarray_1d, bins= bin_num, range=[imarray_1d.min(), imarray_1d.max()])
    
    return(histogram)