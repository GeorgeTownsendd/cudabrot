import pygame
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer

MAX_DEPTH = 256

pygame.init()

@jit
def mandel(x, y, max_iters):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


mandel_gpu = cuda.jit(uint32(f8, f8, uint32), device=True)(mandel)


@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


@cuda.jit((f8, f8, f8, f8, uint8[:,:], uint32))
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = cuda.grid(2)
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y
      image[y, x] = mandel_gpu(real, imag, iters)


def rectangle_zoom(l, t, width, height, t_l, t_t, scale_factor):
    new_l = (scale_factor * (l - t_l)) + t_l
    new_t = (scale_factor * (t - t_t)) + t_t

    new_width = width * scale_factor
    new_height = height * scale_factor

    new_r = new_l + new_width
    new_b = new_t + new_height

    return new_l, new_r, new_b, new_t


size = np.array((1024, 1536)) * 1
display = pygame.display.set_mode(size[::-1])

#t, b, l, r = -2.0, 1.0, -1.0, 1.0
l, r, b, t = -2.0, -1.7, -0.1, 0.1

zoom_factor = 2
zoom_n = 1

movex, movey = 0, 0
speed = 0.01

running = True
frame_n = 0

start_time = timer()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                movey = -speed
            elif event.key == pygame.K_DOWN:
                movey = speed
            elif event.key == pygame.K_LEFT:
                movex = -speed
            elif event.key == pygame.K_RIGHT:
                movex = speed


        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                movey = 0
            elif event.key == pygame.K_DOWN:
                movey = 0
            elif event.key == pygame.K_LEFT:
                movex = 0
            elif event.key == pygame.K_RIGHT:
                movex = 0

        elif event.type == pygame.MOUSEWHEEL:
            print(event.y)
            if event.y == 1:
                width = abs(l-r)
                height = abs(t-b)

                t_l, t_t = np.mean([l, r]), np.mean([t, b])

                l, r, b, t = rectangle_zoom(l, t, width, height, t_l, t_t, 1/zoom_factor)
                speed = speed * 1/zoom_factor

            if event.y == -1:
                width = abs(l-r)
                height = abs(t-b)

                t_l, t_t = np.mean([l, r]), np.mean([t, b])

                l, r, b, t = rectangle_zoom(l, t, width, height, t_l, t_t, zoom_factor)
                speed = speed * zoom_factor



    gimage = np.zeros(size, dtype=np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)

    aspect_ratio = size[0] / size[1]

    l += movex
    r += movex

    b += movey
    t += movey

    start_time = timer()
    d_image = cuda.to_device(gimage)
    mandel_kernel[griddim, blockdim](l, r, b, t, d_image, MAX_DEPTH)
    gimage = d_image.copy_to_host()

    surf = pygame.surfarray.make_surface(gimage.T)

    cmap = plt.get_cmap("inferno", MAX_DEPTH)
    colors = cmap(np.linspace(0, 1, MAX_DEPTH)) * 255

    surf.set_palette(colors)

    display.blit(surf, (0, 0))
    pygame.display.update()
    frame_n += 1

end_time = timer()
runtime = end_time - start_time
fps = runtime / frame_n

print('{} s, {} frames'.format(runtime, frame_n))
