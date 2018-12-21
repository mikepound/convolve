# convolve
Simple demonstration of separable convolutions. I've written this in python 3 using cython for a significant speed boost. This is a demo project only, it could contain errors! In most cases if you want to use convolutions you're safer using a library like numpy or opencv. These tend to use a fourier transform for large convolutions, so I avoided them here just to show the performance benefits.

This is written in python, but I may well add a C# implementation soon, since getting cython compiled in windows is a little more work.

## Python
To run this code you'll need python 3, and cython installed. I've used this in linux, it should port fairly easily to other operating systems. Cython in windows is a little more difficult, but doable!

### Compiling with Cython

Cython compiles your (almost) python into C, which you then compile into a python module. You can compile using cython like this:

`cython -a compute.pyx`

The -a flag outputs a nice HTML file showing where C and Python interact. I've optimised this code for C, so there aren't many yellow lines. It was a lot worse when I started!

This will produce a c file, which you compile using something like this:

`gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.6m -o compute.so compute.c`

Notice you need to point the compiler ar your installed python headers - which may not be in the same place as mine.

If compilation is sucessful, then you'll obtain a compute.so file, which can be imported from python as normal. You can

### Running the code
You can run the code like this:

`python run.py --sigma 3.0 ./christmas.jpg ./christmas.out.jpg`

My included christmas tree picture is fairly large, so even a small convolution will take a while. You'll see quite a difference if you run this with the `--no_separable_filters` option though.






