import pygame
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer
import time

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

'''
mandel_gpu = cuda.jit(uint32(f8, f8, uint32), device=True)(mandel)

@jit
def julia(x, y, a, b, max_iters):
    c = complex(x, y)
    z = complex(a, b)
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters
'''


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
    new_b = new_t + new_height

    return new_l, new_r, new_b, new_t


class FractalWindow:
    def __init__(self, fractal_func, width, height, xy, ab, window_size=(1024, 1536)):
        self.fractal_func = fractal_func
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
        self.speed = 0.01
        self.zoom_factor = 2
        self.zoom_n = 1
        self.size = window_size
        self.zoom_changed = True

    def tick(self):
        if self.movex != 0 or self.movey != 0 or self.zoom_changed or self.new_fractal_func:
            self.l += self.movex
            self.r += self.movex

            self.b += self.movey
            self.t += self.movey

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


mandel_frame = FractalWindow(julia_kernel, 1024, 2048, (0, 0), (0, 0), window_size=(1024, 1024))
mandel_frame2 = FractalWindow(julia_kernel, 1024, 2048, (1024, 0), (0, 0), window_size=(1024, 1024))
frames = [mandel_frame, mandel_frame2]

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
            mandel_frame2.update_fractal_func(mouse_coords)
    else:
        highlighted_frame = mandel_frame2
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
                mandel_frame2.update_fractal_func(mouse_coords)
            elif event.button == 2:
                update_right = not update_right

    mandel_frame.tick()
    mandel_frame2.tick()
    pygame.display.update()
    frame_n += 1

end_time = timer()
runtime = end_time - start_time
fps = runtime / frame_n

print('{} s, {} frames'.format(runtime, frame_n))
