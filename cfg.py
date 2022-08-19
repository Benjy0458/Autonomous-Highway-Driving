# Pygame window dimensions
window_width = None # If None, defaults to the width of the screen
window_height = 120

FPS = 30 # Sets the frame rate
simulation_time = 1200 # Length of time to run the simulation for in seconds
time_gain = 1 # The ratio between simulation time and real time
traffic_density = 1 # Vehicle spawns per second
seed = None # Set a seed (integer) to obtain repeatable results from the simulation. If None the current time is used.
hybrid = False # Toggle between normal and hybrid Astar search
IDM = True # Toggle between PID and IDM controllers

running = True # The simulation ends if this is False

# Agent details
agent_length = 15 # Length
agent_width = 8 # Width
INIT_TRACK = 5 # Start lane

# Lanes contains the y-values for each lane.
lanes = {
    0: 19,
    1: 32,
    2: 44,
    3: 75,
    4: 89,
    5: 102
}

# lane_velocities contains the default velocity corresponding to each lane in mph
lane_velocities = {
    0: -50,
    1: -60,
    2: -70,
    3: 70,
    4: 60,
    5: 50,
}
