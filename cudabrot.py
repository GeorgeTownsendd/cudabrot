import pygame
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer
import time

from least_squares_example import *

MAX_DEPTH = 256

pygame.init()


def poor_mans_ls_fractal(min_x, max_x, min_y, max_y, width, height, bit_depth=8, resolution_downscale=32):
    print(min_x, max_x, min_y, max_y)
    x = np.linspace(min_x, max_x, width//resolution_downscale)
    y = np.linspace(min_y, max_y, height//resolution_downscale)

    xx, yy = np.meshgrid(x, y)

    color_stretch = 256 // bit_depth
    zz = perform_ls(xx, yy) * color_stretch
    print(np.max(zz))

    return zz.repeat(resolution_downscale, axis=0).repeat(resolution_downscale, axis=1)

@jit
def mandel(x, y, max_iters):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


@jit
def julia(x, y, max_iters):
    c = complex(x, y)
    z = complex(0, 0)
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


mandel_gpu = cuda.jit(uint32(f8, f8, uint32), device=True)(mandel)
julia_gpu = cuda.jit(uint32(f8, f8, uint32), device=True)(julia)
#ls_gpu = cuda.jit(uint32(f8, f8, uint32), device=True)(perform_ls)

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


@cuda.jit((f8, f8, f8, f8, uint8[:,:], uint32))
def julia_kernel(min_x, max_x, min_y, max_y, image, iters):
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
            image[y, x] = julia_gpu(real, imag, iters)


def rectangle_zoom(l, t, width, height, t_l, t_t, scale_factor):
    new_l = (scale_factor * (l - t_l)) + t_l
    new_t = (scale_factor * (t - t_t)) + t_t

    new_width = width * scale_factor
    new_height = height * scale_factor

    new_r = new_l + new_width
    new_b = new_t - new_height

    print(new_l, new_r, new_b, new_t)

    return new_l, new_r, new_b, new_t


class FractalWindow:
    def __init__(self, fractal_func, width, height, xy, ab, window_size=(1024, 1536), cuda_enabled=True, resolution_downscale=32):
        self.fractal_func = fractal_func
        self.cuda_enabled = cuda_enabled

        self.width = width
        self.height = height
        self.xy = xy
        self.ab = ab

        self.l = l
        self.r = r
        self.b = b
        self.t = t

        self.movex = 0
        self.movey = 0
        self.speed = 0.05
        self.zoom_factor = 2
        self.zoom_n = 1
        self.size = window_size
        self.resolution_downscale = resolution_downscale
        self.zoom_changed = True

    def tick(self):
        if self.movex != 0 or self.movey != 0 or self.zoom_changed or self.new_fractal_func:
            self.l += self.movex
            self.r += self.movex

            self.b += self.movey
            self.t += self.movey

            if self.cuda_enabled:
                gimage = np.zeros(self.size, dtype=np.uint8)
                blockdim = (32, 8)
                griddim = (32, 16)

                copy_start = time.time()
                d_image = cuda.to_device(gimage)

                processing_start = time.time()
                self.fractal_func[griddim, blockdim](self.l, self.r, self.b, self.t, d_image, MAX_DEPTH)
                processing_end = time.time()
                self.last_frame_time = processing_end - processing_start

                gimage = d_image.copy_to_host()
                copy_end = time.time()
                self.last_full_time = copy_end - copy_start

                self.surf = pygame.surfarray.make_surface(gimage.T)
            else:
                copy_start = time.time()
                processing_start = time.time()
                print(self.resolution_downscale)
                image = poor_mans_ls_fractal(self.l, self.r, self.b, self.t, self.width, self.height, resolution_downscale=self.resolution_downscale)
                processing_end = time.time()
                copy_end = time.time()
                self.last_frame_time = processing_end - processing_start
                self.last_full_time = copy_end - copy_start


                self.surf = pygame.surfarray.make_surface(image)

            self.surf.set_palette(colors)
            self.zoom_changed = False
            self.new_fractal_func = False

        display.blit(self.surf, self.xy)

        font = pygame.font.SysFont(None, 24)

        render_time = font.render('Render time: {} (including copy), {} (GPU render only)'.format(round(self.last_full_time, 5), round(self.last_frame_time, 5)), True, (0, 255, 0))
        fps_indicator = font.render('FPS: {} (including copy), {} (GPU render only)'.format(round(1/self.last_full_time), round(1/self.last_frame_time)), True, (0, 255, 0))

        display.blit(render_time, self.xy + np.array([20, 20]))
        display.blit(fps_indicator, self.xy + np.array([20, 40]))

    def zoom(self, steps):
        for i in range(abs(steps)):
            i *= 1.2
            cwidth = abs(self.l-self.r)
            cheight = abs(self.t-self.b)
            t_l, t_t = np.mean([self.l, self.r]), np.mean([self.t, self.b])

            if steps > 0:
                self.l, self.r, self.b, self.t = rectangle_zoom(self.l, self.t, cwidth, cheight, t_l, t_t, 1 / self.zoom_factor)
                self.speed = self.speed * 1/self.zoom_factor
            elif steps < 0:
                self.l, self.r, self.b, self.t = rectangle_zoom(self.l, self.t, cwidth, cheight, t_l, t_t, self.zoom_factor)
                self.speed = self.speed * self.zoom_factor

        self.zoom_n += steps
        self.zoom_changed = True

    def coordinate_from_mouse(self, xy):
        xc_range = abs(self.l - self.r)
        yc_range = abs(self.t - self.b)

        px, py = xy

        rel_x = xc_range * (px / (size[1] - self.width))
        rel_y = yc_range * (py / (size[0] - self.height))

        xt, yt = self.l + rel_x, self.t + rel_y

        return (xt, yt)

    def update_fractal_func(self, mouse_coords):
        #self.l, self.r, self.b, self.t = -2.0, 1.0, -1.0, 1.0

        @jit
        def new_julia(x, y, max_iters):
            c = complex(x, y)
            z = complex(mouse_coords[0], mouse_coords[1])
            c, z = z, c
            for i in range(max_iters):
                z = z * z + c
                if (z.real * z.real + z.imag * z.imag) >= 4:
                    return i

            return max_iters

        new_julia_gpu = cuda.jit(uint32(f8, f8, uint32), device=True)(new_julia)

        @cuda.jit((f8, f8, f8, f8, uint8[:, :], uint32))
        def new_julia_kernel(min_x, max_x, min_y, max_y, image, iters):
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
                    image[y, x] = new_julia_gpu(real, imag, iters)

        self.fractal_func = new_julia_kernel
        self.new_fractal_func = True

    def get_extent(self):
        return self.l, self.b, self.r, self.t



size = np.array((1024, 2048)) * 1
display = pygame.display.set_mode(size[::-1])

cmap = plt.get_cmap('plasma', MAX_DEPTH)
colors = cmap(np.linspace(0, 1 * (MAX_DEPTH//256), MAX_DEPTH)) * 255
colors[:,-1] = 255

l, r, b, t = -2.0, 1.0, -1.5, 1.5
#l, r, b, t = -2.0, -1.7, -0.1, 0.1

l, r, b, t = -2.0, 1.0, -1.5, 1.5
mandel_frame = FractalWindow(julia_kernel, 1024, 2048, (0, 0), (0, 0), window_size=(1024, 1024))

l, r, b, t = -100, 100, -100, 100
ls_frame = FractalWindow(perform_ls, 1024, 2048, (1024, 0), (0, 0), window_size=(1024, 1024), cuda_enabled=False)


frames = [mandel_frame, ls_frame]

update_right = False

running = True
frame_n = 0

start_time = timer()
while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_coords = mandel_frame.coordinate_from_mouse(mouse_pos)
    if mouse_pos[0] < 1024:
        highlighted_frame = mandel_frame
        if update_right:
            ls_frame.update_fractal_func(mouse_coords)
    else:
        highlighted_frame = ls_frame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                highlighted_frame.movey = -highlighted_frame.speed
            elif event.key == pygame.K_DOWN:
                highlighted_frame.movey = highlighted_frame.speed
            elif event.key == pygame.K_LEFT:
                highlighted_frame.movex = -highlighted_frame.speed
            elif event.key == pygame.K_RIGHT:
                highlighted_frame.movex = highlighted_frame.speed

            elif event.key == pygame.K_PAGEUP:
                if not highlighted_frame.cuda_enabled:
                    highlighted_frame.resolution_downscale //= 2
                    highlighted_frame.zoom_changed = True
            elif event.key == pygame.K_PAGEDOWN:
                if not highlighted_frame.cuda_enabled:
                    highlighted_frame.resolution_downscale *= 2
                    highlighted_frame.zoom_changed = True

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                highlighted_frame.movey = 0
            elif event.key == pygame.K_DOWN:
                highlighted_frame.movey = 0
            elif event.key == pygame.K_LEFT:
                highlighted_frame.movex = 0
            elif event.key == pygame.K_RIGHT:
                highlighted_frame.movex = 0



        elif event.type == pygame.MOUSEWHEEL:
            zoom_steps = event.y
            highlighted_frame.zoom(zoom_steps)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                print(mouse_coords)
                ls_frame.update_fractal_func(mouse_coords)
            elif event.button == 2:
                update_right = not update_right

    mandel_frame.tick()
    ls_frame.tick()
    pygame.display.update()
    frame_n += 1

end_time = timer()
runtime = end_time - start_time
fps = runtime / frame_n

print('{} s, {} frames'.format(runtime, frame_n))
