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
    return mat

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
    image_array = image_array.swapaxes(-2,-1)
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
