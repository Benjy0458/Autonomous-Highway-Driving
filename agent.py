# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:35:57 2022
V2:
- The follow_car method now obeys the 2 second rule recommended by the highways agency.
As a result slightly more collisions are detected from behind since npc vehicles do not adjust their speed.
- The conditions for a left lane change have been toughened to reduce erratic behaviour and risky overtakes.
- A deadzone has been added (indicated by the cyan line) so that collisions caused by spawning vehicles are ignored
by the counter.
- The on_event method of the FSM now takes the current lane of the agent as an additional parameter. Hence, right lane
changes are no longer attempted if in the far right lane, for example.

V3: The lateral velocity of the agent during lane changes is now scaled by the lane_change_speed property.

V4:
- Output to the pygame window has been moved to the test03 script.
- Car_follow method has been reworked and is no longer a function of the agent velocity. The acceleration is calculated
in m/s, and converted to mph/s to update the agent velocity.

V5:
- The car follow method now uses PID control to determine the required acceleration. Two parameters are tracked:
The front car distance and the front car velocity.

V7:
- Global variables are now imported from the cfg module.
- Added support for hybrid A* search algorithm.
- Toggle between search algorithms using the SPACE key.


@author: benjy
"""

import pygame
import numpy as np

# State machine:---
from highway_drive import HighwayDrive

# Trajectory planning:----
from Astar_algorithm import Astar
from Hybrid_Astar_algorithm import HybridAstar
from scipy import interpolate as itp
from collections import deque

import cfg

class AgentCar:
    """Car class for the Finite State Machine"""

    min_speed = 30 # minimum speed of the agent car mph
    max_speed = 77 # maximum speed mph
    acceleration = 3 # The maximum acceleration of the agent m/s^2
    deceleration = 9.81 # max deceleration m/s^2
    sensor_range = 250 # maximum range of the sensors of the car m

    def __init__(self, width, height, track, track_length, window=None):
        self.track_length = track_length # Width of the pygame window
        self.init_track = track # initial track
        self.init_speed = cfg.lane_velocities[track] # initial speed
        self.length = width # width of the car
        self.width = height # height
        self.direction = 1 # 1 is right, -1 left
        self.color = (0, 0, 0) # Agent color
        self.track = track
        self.reset() # initial states
        self.rect = pygame.Rect(self.x - int(width / 2), self.y - int(height / 2), self.length, self.width) # rectangle for the car
        self.car_front1 = self.car_front2 = self.track_length if self.direction == 1 else 0
        self.n_hits = self.successes = self.terminal_count = 0 # Counters (number of collisions, reached end of road, total)
        self.deadzone = 100 # Collisions occurring before this distance are not counted
        self.FSM = HighwayDrive() # initialise Finite State Machine

        self.deque = deque(maxlen=1) # Stores the current position of the agent, goal and obstacles for the A* algorithm.
        self.deque_return = deque(maxlen=1) # Stores the shortest path as a tck
        self.Hybrid_A_star = Hybrid_Astar_Generator()
        self.A_star = Astar_Generator()
        self.A_star.send(None)
        self.Hybrid_A_star.send(None)

    def reset(self):
        """Reset the agent."""
        self.x = 0 if self.direction == 1 else cfg.window_width # get the initial x value
        self.y = cfg.lanes[self.track] # get the initial y value
        self.velocity_x, self.velocity_y = cfg.lane_velocities[self.track], 0
        self.color = (0, 0, 0)
        self.n_track_changes = 0
        self.goal_pos = (self.x + 40, self.y)

    def update(self, dt, vehicles, surface = None):
        """update the agent's car each frame.
        dt       = frame time
        vehicles = list of other cars as rectangle objects
        surface  = provide to draw the agent on the window"""
        car_list, velocities = [vehicle.rect for vehicle in vehicles], [vehicle.x_velocity for vehicle in vehicles]

        done = self.collision_detect(car_list) # Check for collisions
        if done: # Reset the agent if crashed or reached the end
            if self.x > self.deadzone: self.terminal_count += 1
            self.reset()  # Reset the current agent parameters (Eg. position, velocity, current rewards etc.).

        # Get the relative distance and velocity of the closest vehicles in each lane.
        distances_rel, velocities_rel, closest_cars = self.radar(car_list, velocities, surface)

        acc = self.follow_car(distances_rel[0], velocities_rel[0], dt) # Calculate the required acc. of the agent.
        self.velocity_x += acc * 2.237 # Update the agent velocity. Convert acc. to mph/s
        if self.max_speed <= self.velocity_x: self.velocity_x, acc = self.max_speed, 0 # Ensure agent velocity is within acceptable range
        elif self.min_speed >= self.velocity_x: self.velocity_x, acc = self.min_speed, 0

        self.finite_state_machine(distances_rel, velocities_rel) # Update FSM and decide whether to perform a lane change

        # A* path search------------------
        obstacles = [car.inflate(self.length, self.width) for car in closest_cars] # Modify obstacle size to account for Agent dimensions

        # Bounds the search area for hybrid A* between the current and goal pos.
        min_x, max_x = self.x, self.goal_pos[0] +1
        min_y, max_y = min(self.y, self.goal_pos[1]), max(self.y, self.goal_pos[1]) +1

        if self.goal_pos[0] > max_x: self.goal_pos = (max_x, self.goal_pos[1])
        new_path = self.Hybrid_A_star.send(((self.x, self.y), self.goal_pos, min_x, max_x, min_y, max_y, obstacles)) if cfg.hybrid else self.A_star.send(((self.x, self.y), self.goal_pos, obstacles)) # Get the solution path
        if new_path: self.path = new_path
        self.x += self.velocity_x * dt  # Update Agent x-position
        try: xs, ys = itp.splev(np.linspace(0, 1, 1000), self.path) # Discrete coordinates for plotting the spline
        except (ValueError, TypeError, UnboundLocalError): pass # Not enough values to unpack, Cannot unpack non-iterable NoneType object, local variable 'path' referenced before assignment
        else:
            [pygame.draw.line(surface, (250, 0, 250) if cfg.hybrid else (0, 250, 250), (xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(len(xs) - 1)]

            # Get the smallest x value greater than the next x position in the solution path and set the corresponding v position.
            next_x = np.where(xs > self.x) # Filter the solution path to contain only x values greater than the next position
            try: x = min(min(next_x))
            except ValueError: pass # Handles min() arg is an empty sequence
            else: self.y = ys[x] # Update Agent y-position
        #---------------------------------------------------------------------------
        self.rect.center = (self.x, self.y) # Update the location of the agent center

        return acc / 2.237

    def finite_state_machine(self, distances_rel, velocities_rel):
        """Handles behaviour relating to the FSM."""
        observation = self.observe_surrounding_vehicles(distances_rel, velocities_rel) # Observe surrounding vehicles
        self.FSM.on_event(observation, self.track) # Update the state of the FSM

        # Lane change logic:--------------
        if str(self.FSM.state) == 'LaneChangeLeftState': # change track left
            """Left lane change only if front vehicle is at least 2/3 the current follow distance in front,
            the separation between vehicles in the left lane is at least the safe follow distance and
            the distance to the vehicle behind in the left lane is at least the safe follow distance."""
            if (distances_rel[2] > distances_rel[0] * 2/3) and (distances_rel[2] + distances_rel[3] > 2 * velocities_rel[2] / 2.237 + 2 * self.length) and distances_rel[3] > 2 * velocities_rel[2] / 2.237:
                goal_pos = (self.x + 0.5 * distances_rel[2], cfg.lanes[self.track - 1])
                self.change_track("l")
            else: goal_pos = (self.x + 0.5 * distances_rel[0], cfg.lanes[self.track])
        elif str(self.FSM.state) == 'LaneChangeRightState': # change track right
            goal_pos = (distances_rel[4] + self.x, cfg.lanes[self.track + 1]) if self.track < 5 else (self.x + 0.5 * distances_rel[0], self.y)
            self.change_track("r")
        elif str(self.FSM.state) == 'FollowState' or str(self.FSM.state) == 'FreeRideState':
            goal_pos = (self.x + 0.5 * distances_rel[0], cfg.lanes[self.track])

        self.goal_pos = goal_pos # Update the goal position for the path search

    def observe_surrounding_vehicles(self, distances_rel, velocities_rel):
        """Returns 1 of 4 states depending on the distance to surrounding vehicles."""
        # if self.counter == 0:  # Don't update the state of the FSM if a lane change maneuver is still taking place.
        if abs(self.y - self.goal_pos[1]) < 2:
            if (distances_rel[0] < 100) and (velocities_rel[0] < self.max_speed - 10):
                observation = 'slow_vehicle'
            elif distances_rel[0] < 100:
                observation = 'vehicle_ahead'
            elif (distances_rel[4] > 200) and (distances_rel[5] > 20):
                observation = 'right_lane_free'
            elif distances_rel[0] >= 100:
                observation = 'clear_road'

            return observation

    def collision_detect(self, car_list):
        """Returns True if the agent is colliding with any vehicles, False otherwise. """
        collide = self.rect.collidelist(car_list)  # Check if the agent is colliding any vehicles

        # If the car from the carlist is colliding with the agent
        if collide != -1:
            self.color = (255, 0, 0)  # change the color to red
            done = True  # Finish the frame
            if self.x > self.deadzone: self.n_hits += 1 # Do not penalise collisions where an npc spawns on top of the agent.
        elif self.x >= self.track_length:  # Finish the frame the agent has reached the end of the road.
            self.successes += 1
            done = True
        else:
            self.color = (0, 0, 0)
            done = False

        return done

    def follow_car(self, front_car_distance, front_car_velocity, dt):
        """Calculates the required acceleration to maintain a safe distance to the lead vehicle."""
        def PID():
            s_safe = 2 * front_car_velocity / 2.237 # Following 2 second rule. Velocities in m/s
            s_prime = max(s_safe, 3)  # The desired distance to the front vehicle (defaults to 3m if agent is stationary).

            # PID control
            distance_control, velocity_control = self.PID(0.00001, 0.01, 0.0005, dt), self.PID(0.00001, 0.01, 0.0005, dt) # PID controllers
            distance_control.send(None), velocity_control.send(None) # Initialise generator functions

            acc1 = distance_control.send((s_prime, front_car_distance)) # PV and SP are swapped to ensure acc has correct sign.
            acc2 = velocity_control.send((self.velocity_x / 2.237, front_car_velocity / 2.237))
            acc = acc1 + acc2 # Net control action is sum of distance and velocity controller actions.

            if acc > self.acceleration: acc = self.acceleration
            elif acc < -self.deceleration: acc = -self.deceleration

            return acc

        def IDM():
            IDM_control = self.IDM(self.max_speed / 2.237, self.acceleration)
            IDM_control.send(None)
            return IDM_control.send((self.velocity_x / 2.237, front_car_distance, front_car_velocity / 2.237))

        return IDM() if cfg.IDM else PID()

    def change_track(self, track):
        """Change the track of the agent car."""
        # if not self.counter:
        if self.y != self.goal_pos[1]:
            if track == "l":    # change to left track
                if self.track > 3:
                    self.track -= 1 * self.direction
                    self.n_track_changes += 1
            elif track == "r": # change track to right track
                if self.track < 5:
                    self.track += 1 * self.direction
                    self.n_track_changes += 1

    def radar(self, car_list, velocities, surface = None):
        """Returns a list of the distance and velocity of the closest vehicle (front and behind) in each lane.
            [0],[1] - front, behind in same lane as agent.
            [2], [3] - front, behind in left lane.
            [4], [5] - front, behind in right lane.
        Each list in car_lists contains tuples with the vehicle dimensions and its velocity.
        car_list[0/1] contains vehicles in front in the agent's lane.
        car_list[2/3] '' left lane relative to agent.
        car_list[4/5] '' right lane relative to agent.
        Odd index means the vehicle is behind the agent. """
        car_lists = [[], [], [], [], [], []]
        cars, vels = (c for c in car_list), (v for v in velocities)
        for car, velocity in zip(cars, vels): # go through each car
            track = AgentCar.get_track(car) # what lane is the car in?
            if track == self.track: car_lists[0].append((car, velocity)) if car.left > self.x else car_lists[1].append((car, velocity)) # if the car is in the same lane as the agent
            elif track == self.track - 1: # if car is in the left lane (relative to agent)
                if self.direction == 1 and track == 2: continue # ignore cars between 2 main lanes
                car_lists[2].append((car, velocity)) if car.center[0] > self.x else car_lists[3].append((car, velocity)) # Check if car is in front of the agent
            elif track == self.track + 1: # if car is in the right lane
                if self.direction == -1 and track == 3: continue # ignore cars between 2 main lanes
                if car.center[0] > self.x: car_lists[4].append((car, velocity)) # if the car is ahead of the agent
                else: car_lists[5].append((car, velocity)) # if the car is behind the agent

        distances_rel, velocities_rel, closest_cars = [], [], []
        # get distances and draw rays (if surface is provided)
        for i in range(6): # for each track and direction in the car list
            # Draws the appropriate ray in the window
            c, distance, velocity = self.get_closest_car_track(car_lists[i], False if i % 2 else True) # Find the closest vehicle
            if c is not None: closest_cars.append(c)
            if not i % 2:
                if c is not None:
                    if self.direction == 1:
                        if i == 2: self.car_front1 = c.left
                        if i == 4: self.car_front2 = c.left
                    else:
                        if i == 3: self.car_front1 = c.right
                        if i == 5: self.car_front2 = c.right

                    if surface is not None: self.draw_ray(surface, c, side=False if distance > 0 else True)
            else:
                if c is not None and surface is not None:
                    self.draw_ray(surface, c, pos=False) if distance > 0 else self.draw_ray(surface, c, pos=False, side=True)

            distances_rel.append(distance), velocities_rel.append(velocity) # Append the relative distance and velocity of the current vehicle to the corresponding list.

        return distances_rel, velocities_rel, closest_cars

    def get_closest_car_track(self, car_list, pos=True):
        """Finds the closest car in the car list.
        Returns the dimensions of the car, the relative distance, and its velocity.
        If the car list is empty or the vehicle is out of range of the sensor, car is None. """
        if car_list: # if the list is not empty
            l = [] # list to store distance to all the cars
            for car, _ in car_list: # go through each car
                k = self.length / 2 + car.width / 2 + 5 # the distance between the midpoints of the vehicles required for a 5m gap
                # if the car is very close to the agent append 0
                if pos: l.append(car.left - self.x) if car.center[0] - self.x > k else l.append(0) # for the positive x direction
                else: l.append(self.x - car.right) if self.x - car.center[0] > k else l.append(0) # for the negative direction

            # get the minimum length
            i = np.argmin(l)
            car = car_list[i][0]

            distance = car.left - self.x if pos else self.x - car.right # calculate distance
            if distance > AgentCar.sensor_range: return None, AgentCar.sensor_range, 0.0 # if the car is out of the range

            return car, distance, car_list[i][1]

        return None, AgentCar.sensor_range, 0.0 # If the car list is empty

    def draw_ray(self, surface, car, side=False, pos=True):
        """Draw a ray to closest car."""
        track = AgentCar.get_track(car)
        if not side: pygame.draw.line(surface, (255, 0, 0), (self.x, cfg.lanes[track]), (car.left if pos else car.right, cfg.lanes[track]), 2)
        else: pygame.draw.line(surface, (0, 255, 255), (self.x, self.y), (self.x, cfg.lanes[track]), 2)

    def draw(self, surface):
        """Draw the agent."""
        pygame.draw.rect(surface, self.color, self.rect)

    @staticmethod
    def get_track(rect):
        """Get the track of the given rectangle car."""
        temp = np.array(list(cfg.lanes.values()))
        return int(np.argmin(np.abs(temp - rect.center[1])))

    @staticmethod
    def PID(Kp, Ki, Kd, dt, MV_bar=0):
        # Initialise stored data
        e_prev = 0  # Previous error value
        I = 0  # Integral value
        MV = MV_bar  # Initial control
        while True:
            PV, SP = yield MV # yield MV, wait for new PV, SP

            # PID calculations
            e = SP - PV  # Tracking error
            P = Kp * e  # Proportional term
            I = I + Ki * e * dt  # Integral term
            D = Kd * (e - e_prev) / dt  # Derivative term

            MV = MV_bar + P + I + D # Controller action is sum of 3 terms
            e_prev = e # Update stored data for next iteration

    @staticmethod
    def IDM(desired_velocity, max_acceleration, acc_bar=0):
        desired_velocity = desired_velocity  # The velocity the vehicle would drive at in free traffic
        minimum_spacing = 3  # The minimum desired net distance. (Car can't move if distance to the car in front isn't greater than this.)
        time_headway = 1  # The minimum possible time to the vehicle in front
        max_acc = max_acceleration  # The max vehicle acceleration
        delta = 4  # Acceleration exponent
        acc = acc_bar # Initial control
        while True:
            velocity_x, front_car_distance, front_car_velocity = yield acc
            s_star = minimum_spacing + velocity_x * time_headway
            acc = max_acc * (1 - (velocity_x / desired_velocity) ** delta - (s_star / front_car_distance) ** 2)

def Astar_Generator():
    WIDTH, HEIGHT = cfg.window_width, cfg.window_height  # Width, height of Pygame window
    rows, cols = int(0.5 * HEIGHT), int(0.1 * WIDTH)  # rows, cols should be factors of HEIGHT, WIDTH respectively. Sets the grid density, Number of rows/columns in grid
    A_star = Astar(WIDTH, HEIGHT, rows, cols)  # Initialise the Astar algorithm
    path = None
    while True:
        agent_pos, goal_pos, obstacles = yield path  # Wait for new data
        if agent_pos is not None and goal_pos is not None and obstacles is not None:
            try: path = A_star.run(agent_pos, goal_pos, obstacles)  # Run A* algorithm
            except (ValueError, TypeError): continue # Not enough values to unpack, Cannot unpack non-iterable NoneType object

def Hybrid_Astar_Generator():
    WIDTH, HEIGHT = cfg.window_width, cfg.window_height # Width, height of Pygame window
    hy_a_star = HybridAstar(min_x=0, max_x=WIDTH, min_y=0, max_y=HEIGHT, width=WIDTH, height=HEIGHT) # Initialise Hybrid A* algorithm
    path = None
    while True:
        agent_pos, goal_pos, min_x, max_x, min_y, max_y, obstacles = yield path # Wait for new data
        if agent_pos is not None and goal_pos is not None and obstacles is not None:
            try: path = hy_a_star.run((agent_pos[0], agent_pos[1], 0), (goal_pos[0], goal_pos[1], 0), min_x, max_x, min_y, max_y, obstacles=obstacles) # Get the shortest path to the goal
            except (ValueError, TypeError): continue # Not enough values to unpack, Cannot unpack non-iterable NoneType object
