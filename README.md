# convolve
Simple demonstration of separable convolutions. This includes a standard gaussian blur, and a more recent lens blur using complex kernels. This is a demo project only, it could contain errors!

## Gaussian Blur
I've written this in python 3 using cython for a significant speed boost. In most cases if you want to use convolutions you're safer using a library like numpy or opencv. These tend to use a fourier transform for large convolutions, so I avoided them here just to show the performance benefits.

### Python
To run this code you'll need python 3, and cython installed. I've used this in linux, it should port fairly easily to other operating systems. Cython in windows is a little more difficult, but doable!

### Compiling with Cython

Cython compiles your (almost) python into C, which you then compile into a python module. You can compile using cython like this:

`cython -a compute.pyx`

The -a flag outputs a nice HTML file showing where C and Python interact. I've optimised this code for C, so there aren't many yellow lines. It was a lot worse when I started!

This will produce a c file, which you compile using something like this:

`gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.6m -o compute.so compute.c`

Notice you need to point the compiler ar your installed python headers - which may not be in the same place as mine.

If compilation is successful, then you'll obtain a compute.so file, which can be imported from python as normal. Windows compilation will be a little more complex, as gcc isn't available to you. [Here might help!](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows)

### Running the code
You can run the code like this:

`python run.gaussian.py --sigma 3.0 ./christmas.jpg ./christmas.out.jpg`

My included christmas tree picture is fairly large, so even a small convolution will take a while. You'll see quite a difference if you run this with the `--no_separable_filters` option though.

## Lens Blur
This is also written in python 3, without cython this time. Complex convolutions aren't too difficuly, but I decided the scypi library was easier here. This lens blur is the sum of a number of components, which means that the time taken scales linearly with the number of components you wish to use. Larger images with high component counts will take a fair while!

### Running the code
You can run the code like this:

`python run.lens.py --radius 50 --components 3 --exposure_gamma 3 ./M35.jpg ./M35.out.jpg`

I found numpy raises memory issues if you use very large images, avoiding the stack function would probably fix this, but I decided not to worry about it.

One interesting difference between this simulated lens blur and a real lens is that the lighting works differently. I used a gamma-like function to increase the exposure prior to convolving, and then reverse this afterwards, I found this tends to highlight the brighter areas of the image, but results will depend on your settings, and the image you are filtering.



