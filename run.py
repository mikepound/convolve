import time
import numpy as np
from PIL import Image
import argparse
import compute

# Helper function to read an image into a CxHxW numpy float32 array 
def read_image(path):
    img = np.array(Image.open(path))
    arr = np.ascontiguousarray(img.transpose(2,0,1), dtype=np.float32)
    arr /= 255
    return arr

# Reverse helper function to write an image from a CxHxW numpy array
def write_image(img, path):
    img *= 255
    img = img.transpose(1,2,0).astype(np.uint8)
    Image.fromarray(img).save(path)

# Produces a 2D gaussian kernel of standard deviation sigma and size 2*sigma+1
def gaussian_kernel_2d(sigma):
    kernel_radius = np.ceil(sigma) * 3
    kernel_size = kernel_radius * 2 + 1
    ax = np.arange(-kernel_radius, kernel_radius + 1., dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# Produces a 1D gaussian kernel of standard deviation sigma and size 2*sigma+1
def gaussian_kernel_1d(sigma):
    kernel_radius = np.ceil(sigma) * 3
    kernel_size = kernel_radius * 2 + 1
    ax = np.arange(-kernel_radius, kernel_radius + 1., dtype=np.float32)
    kernel = np.exp(-(ax**2) / (2. * sigma**2))
    return (kernel / np.sum(kernel)).reshape(1,kernel.shape[0])

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sigma', nargs='?', type=float, default=3.0)
    parser.add_argument('--no_separable_filters', action='store_true')
    parser.add_argument("input_file", help="The input image file.")
    parser.add_argument("output_file", help="The output image file.")
    
    args = parser.parse_args()
    
    # Read image - I'm using floats to store the image, this isn't necessary but saves casting etc. during convolution
    img = read_image(args.input_file)

    # Create output of the same size
    output = np.zeros(img.shape, dtype=np.float32)

    # Get current time - I believe perf_counter is a python 3 function
    t0 = time.perf_counter()

    if args.no_separable_filters:
        # NxN convolution
        kernel_2d = gaussian_kernel_2d(args.sigma) # You could create your own kernel here!

        # Convolve
        compute.convolve(img, output, kernel_2d)
        
    else:
        # Nx1 -> 1xN convolution
        kernel_1d = gaussian_kernel_1d(args.sigma)

        # We need to store the half convolved intermediate image.
        # You could save time by going img -> output-> img and not allocating this array.
        # Bearing in mind if you do this you can't use img for anything else.
        intermediate = np.zeros(img.shape, dtype=np.float32) 
        
        # Convolve in two passes - we must store and use the intermediate image, don't read from the input both times!
        compute.convolve(img, intermediate, kernel_1d)
        compute.convolve(intermediate, output, kernel_1d.transpose())
    
    # Measure elapsed time
    t1 = time.perf_counter()
    print ("Elapsed: {0:.3f}s".format(t1-t0))
    
    # Save final image
    write_image(output, args.output_file)

if __name__ == "__main__":
    main()
