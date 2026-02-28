import json
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt

class Particle: 
    def __init__ (self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y 
        self.theta = theta
        self.weight = weight

    def move(self, distance, rotation, noise_dist, noise_rot):
        dist = distance + np.random.normal(0, noise_dist)
        rot = rotation + np.random.normal(0, noise_rot)
        
        self.theta += rot
        self.x += dist * np.cos(self.theta)
        self.y += dist * np.sin(self.theta)

#class ParticleFilter: 


def init_particles(num_particles, map_bool, ppm, origin, dims):
    particles = []
    while len(particles) < num_particles:
        # Pick a random pixel
        px = np.random.randint(0, dims['width_px'])
        py = np.random.randint(0, dims['height_px'])
        
        # If it's a white pixel (traversable), convert to meters and spawn
        if map_bool[py, px]:
            world_x = (px / ppm) + origin[0]
            world_y = ((dims['height_px'] - py) / ppm) + origin[1]
            particles.append(Particle(world_x, world_y, np.random.uniform(0, 2*np.pi)))
    return particles

def visualize_particles(particles, map_mask, ppm, origin, dims):
    plt.figure(figsize=(10, 10))
    
    # 1. Define the physical bounds for the image display
    # Extent format: [xmin, xmax, ymin, ymax]
    world_width = dims['width_px'] / ppm
    world_height = dims['height_px'] / ppm
    extent = [origin[0], origin[0] + world_width, origin[1], origin[1] + world_height]
    
    # 2. Display the Map Mask (Grey for walls, White for floor)
    # We use 'origin=upper' because images are indexed from the top-down
    plt.imshow(map_mask, cmap='bone', extent=extent, origin='upper', alpha=0.4)
    
    # 3. Extract particle coordinates
    px = [p.x for p in particles]
    py = [p.y for p in particles]
    
    # 4. Scatter plot the particles
    plt.scatter(px, py, s=2, c='blue', alpha=1, label='Particles')

    # # 5. (Optional) Plot a few orientation arrows
    # # We only plot a subset so it doesn't get too messy
    # for p in particles[::100]: # Every 100th particle
    #    plt.arrow(p.x, p.y, 0.1 * np.cos(p.theta), 0.1 * np.sin(p.theta), color='red', head_width=0.05)

    plt.title("Particle Filter Localization State")
    plt.xlabel("World X (meters)")
    plt.ylabel("World Y (meters)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def load_map_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    ppm = config['resolution']['pixels_per_meter']
    origin = (config['origin']['x_meters'], config['origin']['y_meters'])
    dims = config['dimensions']
    return ppm, origin, dims




ppm, origin, dims = load_map_config('map_mask_metadata.json')

# convert map_mask.png into a boolean array of accessible pixels. True for walkable areas and False for walls. 
map_bool = np.array(Image.open("maps/map_mask.png").convert("L")) > 128

particles = init_particles(500, map_bool, ppm, origin, dims)
visualize_particles(particles, map_bool, ppm, origin, dims)