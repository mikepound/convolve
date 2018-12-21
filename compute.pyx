# cython: language_level=3
# cython: infer_types=True
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve(float[:, :, ::1] image, float[:, :, ::1] output, float[:, ::1] kernel):
    cdef ssize_t image_height, image_width, channel_count
    cdef ssize_t kernel_height, kernel_width, kernel_halfh, kernel_halfw
    cdef ssize_t x_min, x_max, y_min, y_max, x, y, u, v, c
    
    cdef float value, tmp, total

    channel_count = image.shape[0]
    image_height = image.shape[1]
    image_width = image.shape[2]

    kernel_height = kernel.shape[0]
    kernel_halfh = kernel_height // 2
    kernel_width = kernel.shape[1]
    kernel_halfw = kernel_width // 2

    # Do convolution
    for x in range(image_width):
        for y in range(image_height):
            # Calculate usable image / kernel range
            x_min = max(0, x - kernel_halfw)
            x_max = min(image_width - 1, x + kernel_halfw)
            y_min = max(0, y - kernel_halfh)
            y_max = min(image_height - 1, y + kernel_halfh)

            # Convolve filter
            for c in range(channel_count):
                value = 0
                total = 0
                for u in range(x_min, x_max + 1):
                    for v in range(y_min, y_max + 1):
                        tmp = kernel[v - y + kernel_halfh, u - x + kernel_halfw]
                        value += image[c, v, u] * tmp  
                        total += tmp
                output[c, y, x] = value / total