"""
V3:
- Completely overhauled npc vehicle generation.
npc vehicles are now generated by the spawn_vehicles method using several normal dist. functions.
- The repeated_timer class runs in a separate thread and calls the spawn vehicles method at a given time interval.
Initialisation time of the program is vastly improved and no longer dependent on large .csv files.

V4:
- Global variables are now imported from the cfg module.
- The width of the pygame window defaults to the screen width unless specified.
- Pygame window caption displays which search algorithm is being used by the agent.
"""

import os
import pygame
import time
import random
import numpy as np
import threading
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys

import cfg
import agent # Import agent module
import traffic # Import traffic module
from timer import RepeatedTimer

xs, ys, acc_ys = [], [], [] # Initialise plotting variables.

class ScenarioData:
    def __init__(self, x, y, agent):
        self.timestep = 1/cfg.FPS
        self.start_time = 0
        image = pygame.image.load(r'highway.jpg').convert()
        self.image = pygame.transform.scale(image, (x, y)) # Scale the background image
        self.X = x
        self.Y = y
        self.agent = agent # add the agent car
        self.vehicles = [] # Keeps track of the vehicles currently in the window

    def plot_scenario(self, display_surface, queue):
        """Runs the simulation until reached end of frame list or user quits program.
        Returns the number of scenarios simulated, successes and collisions."""
        print('scenario start')
        self.start_time = time.time() # Store the start time of the simulation
        elapsed_time = 0 # Stores the elapsed time of the simulation
        clock = pygame.time.Clock() # Controls the frame rate

        try:
            # Game loop:------------
            while cfg.running:
                elapsed_time = time.time() - self.start_time # Calculate the elapsed time
                self.terminate(elapsed_time, cfg.simulation_time) # Check if the simulation should end

                display_surface.blit(self.image, (0, 0)) # Fill the pygame window with highway image
                self.update_vehicles(display_surface) # Update NPC vehicles
                acc = self.update_agent(display_surface) # Update and draw the agent

                pygame.display.update() # Update the pygame window

                tracks_count = self.lane_distribution() # Returns the number of vehicles in each lane
                queue.put((elapsed_time, self.agent.velocity_x, acc)) # Send current data to the queue for live plot
                xs.append(elapsed_time), ys.append(self.agent.velocity_x), acc_ys.append(acc) # Log Agent data

                pygame.display.set_caption(f"Elapsed time: {round(elapsed_time)}, FPS: {round(clock.get_fps())}, Num Vehicles: {len(self.vehicles)}, Lane dist: {tracks_count}, "
                                           f"Scenarios simulated: {self.agent.terminal_count}, Collisions: {self.agent.n_hits}, "
                                           f"Speed: {round(self.agent.velocity_x)} mph, Current state: {str(self.agent.FSM.state)}, "
                                           f"Algorithm: {'Hybrid A*' if cfg.hybrid else 'Normal A*'}, ACC: {'IDM' if cfg.IDM else 'PID'}") # Update the window caption

                clock.tick(cfg.FPS * cfg.time_gain) # Ensure program maintains desired frame rate

        finally:
            save_data(xs, ys, acc_ys, "AgentProfile.png")  # Save the agent data in a .png file
            print(f"Elapsed time: {round(elapsed_time, 1)}s{os.linesep}"
                  f"Number of collisions: {self.agent.n_hits}{os.linesep}"
                  f"Number of successes: {self.agent.successes}{os.linesep}"
                  f"Scenarios simulated: {self.agent.terminal_count}") # Display results

    def spawn_vehicles(self, max_vehicles, traffic_density):
        """Creates a new vehicle and appends it to the list of vehicles."""
        new_vehicles = [] # Temporary list to store our new vehicles.
        number_of_vehicles = random.randint(0, max_vehicles) # Assign a random number of vehicles to spawn in a range.

        for i in range(number_of_vehicles):
            new_vehicle = traffic.Vehicle(cfg.lane_velocities, cfg.lanes, self.timestep, self.X, traffic_density)  # Create a new vehicle
            for Npc in range(len(new_vehicles)):  # Iterate through the list of new vehicles
                if new_vehicle.lane == new_vehicles[Npc].lane: break # If multiple vehicles being added they must be in different lanes and not collide with existing vehicles.
            else:
                for car in self.vehicles:
                    if car.lane == new_vehicle.lane and (
                            car.rect.x <= new_vehicle.rect.x <= car.rect.x + car.rect.width) or (
                            car.rect.x <= new_vehicle.rect.x + new_vehicle.rect.width <= car.rect.x + car.rect.width):
                        break
                else: new_vehicles.append(new_vehicle)  # Add the vehicle to the list of new vehicles

        self.vehicles.extend(new_vehicles)  # Add the new vehicles to the vehicle list

    def lane_distribution(self):
        """Returns the number of vehicles in each lane."""
        tracks = [vehicle.lane for vehicle in self.vehicles]
        tracks_count = [tracks.count(track) for track in range(6)]

        return tracks_count

    @staticmethod
    def terminate(elapsed_time, end_time):
        """Gets user input and decides whether to terminate the program."""
        if elapsed_time >= end_time: cfg.running = False # Quit the program if reached the time limit.
        # Close the program on user intervention.
        for event in pygame.event.get():
            if event.type == pygame.QUIT: cfg.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: cfg.hybrid = not cfg.hybrid # Toggle search algorithm
                if event.key == pygame.K_k: cfg.IDM = not cfg.IDM # Toggle ACC

    def update_agent(self, display_surface):
        acc = self.agent.update(self.timestep, self.vehicles, display_surface)
        self.agent.draw(display_surface)  # Draw the agent
        pygame.draw.line(display_surface, (0, 255, 255), (self.agent.deadzone, 0), (self.agent.deadzone, cfg.window_height)) # Collisions detected before this line are not counted

        return acc

    def update_vehicles(self, display_surface):
        """Updates the positions of all traffic on the screen. Removes vehicles that have moved off screen."""
        for vehicle in self.vehicles:
            if 0 <= vehicle.x_pos <= self.X:
                vehicle.update_position(0, 0) # Update the vehicle's position
                pygame.draw.rect(display_surface, vehicle.colour, vehicle.rect) # Draw the vehicle on the display window
            else: self.vehicles.remove(vehicle)

def init_pygame(x=None, y=120):
    """Returns a new named pygame window."""
    x_pos, y_pos = 0, 30
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x_pos, y_pos) # Set the position of the window on the screen
    pygame.init()
    if not x: x = pygame.display.Info().current_w
    cfg.window_width = x
    ds = pygame.display.set_mode((x, y))
    pygame.display.set_caption('Image')

    return ds

def animate(i, ax1, ax2, xs, ys, ays, queue):
    new_data = []
    while not queue.empty(): new_data.append(queue.get())

    [(xs.append(row[0]), ys.append(row[1]), ays.append(row[2])) for row in new_data] # Add new data

    while len(xs) > 1500: del xs[0], ys[0], ays[0] # Plot the last 1500 data points

    ax1.clear(), ax2.clear() # Clear axes
    ax1.set_ylim([20, 80]), ax2.set_ylim([-1.5, 1.5]) # Set axis limits
    ax1.plot(xs, ys, c='k', label='vel. (mph)', linewidth=0.2) # Plot data
    ax2.plot(xs, ays, c='r', label='acc. (m/s^2)', linewidth=0.2)
    ax1.set_ylabel('Vel. (mph)', fontsize=10) # Set axes labels
    ax2.set_ylabel('Acc. (m/s^2)', color='red', fontsize=10)
    ax1.set_xlabel('Elapsed time (s)', fontsize=10)
    ax2.grid(False)
    plt.rcParams['font.size'] = 10 # Set tick size

def live_graph(queue, xs, ys, acc_ys, window_width):
    style.use('fivethirtyeight')

    fig = plt.figure("Agent Profile") # Create new figure instance
    fig.set_size_inches(6, 3) # Set figure window size in inches
    thismanager = plt.get_current_fig_manager()

    x = window_width - 375
    y = 188
    thismanager.window.wm_geometry("+%d+%d" % (x, y))

    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.135, top=0.97) # Adjust the relative position of the subplots
    ax1 = fig.add_subplot(1, 1, 1) # Create the axes
    ax2 = ax1.twinx()

    ani = animation.FuncAnimation(fig, animate, interval=1, fargs=(ax1, ax2, xs, ys, acc_ys, queue)) # Update the figure window every ms.
    plt.show() # Display the plots

def save_data(xs, ys, ays, filename):
    """Saves a plot of the velocity and acceleration trace of the agent."""
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.92)  # Adjust the relative position of the subplots

    ax1 = plt.subplot(2,1,1) # Create subplots
    ax2 = plt.subplot(2,1,2)

    ax1.set_ylim([20, 80])  # Set axis limits
    ax2.set_ylim([-10, 5])
    ax1.plot(xs, ys, c='k', label='vel. (mph)', linewidth=0.05)  # Plot data
    ax2.plot(xs, ays, c='r', label='acc. (m/s^2)', linewidth=0.05)
    ax1.set_title('Agent Velocity/Acceleration Profile')
    ax1.set_ylabel('Vel. (mph)', fontsize=10)  # Set axes labels
    ax2.set_ylabel('Acc. (m/s^2)', fontsize=10)
    ax2.set_xlabel('Elapsed time (s)', fontsize=10)
    plt.rcParams['font.size'] = 10  # Set tick size

    plt.savefig(filename, dpi=1200)

if __name__ == '__main__':
    if cfg.seed: random.seed(cfg.seed)

    display_window = init_pygame(x=cfg.window_width, y=cfg.window_height) # Open a new named pygame window
    agent_fsm = agent.AgentCar(width = cfg.agent_length, height = cfg.agent_width, track= cfg.INIT_TRACK, track_length = cfg.window_width, window = display_window) # Initialise the agent vehicle
    scenario_data = ScenarioData(cfg.window_width, cfg.window_height, agent_fsm) # Create a new instance of scenario data incl. the agent

    queue = multiprocessing.Queue() # Stores data for the live graph
    p1 = multiprocessing.Process(target=live_graph, args=(queue, xs, ys, acc_ys, cfg.window_width))
    p1.start()
    # Spawns up to 3 vehicles each second and adds them to scenario_data.vehicles
    rt = RepeatedTimer(1 / (cfg.traffic_density * cfg.time_gain), scenario_data.spawn_vehicles, 3, cfg.traffic_density)  # it auto-starts, no need of rt.start()
    try:scenario_data.plot_scenario(display_window, queue)
    finally:
        p1.kill() # Stop live plotting
        rt.stop()  # Stop the spawn vehicles thread
        pygame.quit()  # Close the pygame window
