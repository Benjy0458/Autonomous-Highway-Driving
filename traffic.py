"""
Game loop:
Spawn a new vehicle at set time interval (could be randomly chosen?)
Add vehicle to the vehicle list.
When vehicle reaches the end of the road, remove it from the vehicle list.

V2: Added lane_bias property which:
- Tries to account for bias in the random.choice function.
- Controls the distribution of vehicles between the top and bottom 3 lanes of the highway
based on the traffic density parameter.

"""

import random
import pygame

class Vehicle:
    # properties
    scale = 10/3 # Scales the size of the vehicles

    def __init__(self, velocities, lanes, timestep, track_length, traffic_density): # lane velocities, lanes
        self.track_length = track_length # Width of the pygame window
        self.lane_bias = 0.04 if traffic_density > 25 else -0.008 * traffic_density + 0.24 # The probability of a vehicle spawning in the top three lanes. accounts for bias of random.choice towards lanes 0,1,2 with higher values of traffic density
        self.dt = timestep
        self.acceleration = 0 # The initial acceleration of the vehicle
        self.width = random.normalvariate(2, 0.15) * self.scale # Width of the vehicle (y-direction)
        if self.width / self.scale > 2.15: # For spawning trucks
            self.length = random.normalvariate(16.0, 2) * self.scale # For trucks
            self.lane = random.choice([4, 5]) if random.random() > self.lane_bias else random.choice([0, 1]) # Trucks won't spawn in the fast lane
        else:
            self.length = random.normalvariate(4.5, 0.2) * self.scale # Length of the vehicle (x-direction)
            self.lane = random.choice([3,4,5]) if random.random() > self.lane_bias else random.choice([0,1,2]) # Choose a random lane for the vehicle.

        self.x_velocity = velocities[self.lane]  # x velocity of the vehicle
        self.y_velocity = 0
        self.y_pos = lanes[self.lane]  # The y-position of the vehicle's lane

        if self.lane > 2: # Set the start x position depending on travel direction
            self.x_pos = 0
        else:
            self.x_pos = self.track_length

        self.colour = pygame.Color(int(random.random() * 256), int(random.random() * 256),
                                   int(random.random() * 256))  # Assign the vehicle a random colour
        self.rect = pygame.Rect(0, 0, self.length, self.width)  # Create the vehicle's rectangle
        self.rect.center = (self.x_pos, self.y_pos)

    def update_position(self, x_acceleration, y_acceleration):
        self.x_pos += self.x_velocity * self.dt + 0.5 * x_acceleration * self.dt ** 2  # increment position values: s = ut + 0.5at^2
        self.y_pos += self.y_velocity * self.dt + 0.5 * y_acceleration * self.dt ** 2

        self.x_velocity += x_acceleration * self.dt # increment the velocity values: v = u + at
        self.y_velocity += y_acceleration * self.dt

        self.rect.center = (self.x_pos, self.y_pos) # Update the location of the vehicle rectangle
