import time
import numpy as np
from PIL import Image
import argparse
import math
from complex_kernels import *
from scipy import signal
from functools import reduce

def read_image(path):
    img = np.array(Image.open(path))
    arr = np.ascontiguousarray(img.transpose(2,0,1), dtype=np.float32)
    arr /= 255
    return arr

def write_image(img, path):
    img *= 255
    img = img.transpose(1,2,0).astype(np.uint8)
    Image.fromarray(img).save(path)

def gamma_exposure(img, gamma):
    np.power(img, gamma, out=img)

def gamma_exposure_inverse(img, gamma):
    np.clip(img, 0, None, out=img)
    np.power(img, 1.0/gamma, out=img)
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--radius', nargs='?', type=int, default=32)
    parser.add_argument('--components', nargs='?', type=int, default=2)
    parser.add_argument('--exposure_gamma', nargs='?', type=float, default=3.0)
    parser.add_argument("input_file", help="The input image file.")
    parser.add_argument("output_file", help="The output image file.")
    
    args = parser.parse_args()
    
    # Read image - I'm using floats to store the image, this isn't necessary but saves casting etc. during convolution
    img = read_image(args.input_file)

    # Create output of the same size
    output = np.zeros(img.shape, dtype=np.float32)

    # Get current time - I believe perf_counter is a python 3 function
    t0 = time.perf_counter()

    # Obtain component parameters / scale values
    radius = args.radius
    parameters, scale = get_parameters(component_count = args.components)

    # Create each component for size radius, using scale and other component parameters
    components = [complex_kernel_1d(radius, scale, component_params['a'], component_params['b']) for component_params in parameters]

    # Normalise all kernels together (the combination of all applied kernels in 2D must sum to 1)
    normalise_kernels(components, parameters)

    # Increase exposure to highlight bright spots
    gamma_exposure(img, args.exposure_gamma)
    
    # Process RGB channels for all components
    component_output = []
    for component, component_params in zip(components, parameters):
        channels = []
        for channel in range(img.shape[0]):
            inter = signal.convolve2d(img[channel], component, boundary='symm', mode='same')
            channels.append(signal.convolve2d(inter, component.transpose(), boundary='symm', mode='same'))

        # The final component output is a stack of RGB, with weighted sums of real and imaginary parts
        component_image = np.stack([weighted_sum(channel, component_params) for channel in channels])
        component_output.append(component_image)

    # Add all components together
    output_image = reduce(np.add,component_output)

    # Reverse exposure
    gamma_exposure_inverse(output_image, args.exposure_gamma)

    # Avoid out of range values - generally this only occurs with small negatives
    # due to imperfect complex kernels
    np.clip(output_image, 0, 1, out=output_image)

    # Measure elapsed time
    t1 = time.perf_counter()
    print ("Elapsed: {0:.3f}s".format(t1-t0))
    
    # Save final image
    write_image(output_image, args.output_file)

if __name__ == "__main__":
    main()

