"""
V8:
- Obstacle collisions are now detected using the built-in collidepoint function in pygame.
Slightly slower than previous method BUT:
Allows larger timestep to be used before encountering obstacle jumping, hence good performance achieved with
significantly faster runtimes.
Much more accurate for large obstacles.
"""
import heapq as hq
import numpy as np
import inspect
import pygame
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy import interpolate as itp
from operator import itemgetter
from itertools import repeat
from itertools import count
from timeit import default_timer as timer

import cfg

class HybridAstar:
    def __init__(self, min_x, max_x, min_y, max_y, width, height):
        self.width, self.height = width, height # Width, height of the pygame window
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y # Bounds of the search area
        self.l_f, self.l_r = 2, 2.5 # Vehicle data: Front/Rear semi-wheelbase
        self.dt = 4 # Timestep

        self.v = np.linspace(1, 1, num=1) # Directions
        self.delta = np.array([np.deg2rad(np.linspace(-45, 45, num=33))]).reshape(-1,1) # Possible steering angles (num must be odd to have option of delta=0)

        self.reset()

    def reset(self):
        self.open_heap = []  # element of this list is like (cost,node_d)
        self.open_dict = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))
        self.closed_dict = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))
        self.interp_path = () # Stores the tck tuple containing the interpolated shortest path.
        # self.f_current = 0 # Stores the total cost of the current node
        self.count = count() # Count number of times through the while loop

    def create_obstacles(self, vehicle_list):
        """Accepts a list of pygame rectangle objects. Returns a numpy array of (x, y) points. Each rectangle contains 9 points
        points = ['topleft', 'bottomleft', 'topright', 'bottomright', 'midtop', 'midleft', 'midbottom', 'midright', 'center']"""
        # x_obs, y_obs = [], []  # Empty lists to store obstacle x,y values
        points = {23, 24, 25, 31, 32, 33, 34, 38, 39}  # Index of inbuilt points in pygame.rect
        locations = (inspect.getmembers(vehicle, lambda a: not (inspect.isroutine(a))) for vehicle in vehicle_list) # Get the list of attributes associated with the vehicle rects.
        points_list = (veh[point][1] for veh in locations for point in points) # Get all the discrete points of the pygame rects.
        xy_obs = [point for point in points_list]

        return np.array(xy_obs)

    def run(self, start_pos, goal_pos, min_x, max_x, min_y, max_y, obstacles):
        """Runs the Hybrid A* algorithm. obstacles is a list of pygame rect objects."""
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y
        a = goal_pos # (x, y)
        b = (self.width, self.height) # (x, y)
        if all(ai <= bi for ai, bi in zip(a, b)): # Make sure the goal position is within the bounds of the pygame window.
            self.reset() # Reset the state of all nodes and their corresponding costs.
            self.start_pos, self.goal_pos = start_pos, goal_pos # x,y start and goal position.
            self.obstacles = self.create_obstacles(obstacles) # List of (x, y) values of obstacles
            self.vehicle_list = obstacles # List of pygame rects of obstacles

            self.start = np.array((self.start_pos[0], self.start_pos[1], np.deg2rad(self.start_pos[2] % 360))) # (x_start, y_start, theta_start)
            self.goal = np.array((self.goal_pos[0], self.goal_pos[1], np.deg2rad(self.goal_pos[2] % 360))) # (x_goal, y_goal, theta_goal)

            f_start = self.h(self.start[:2], self.goal[:2]) # Total cost of the start node is just the heuristic cost

            start = self.start
            hq.heappush(self.open_heap, (f_start, tuple(start))) # Put the start node and cost into the heapq
            self.open_dict[tuple(start)] = (f_start, start, (start, start), 1, 0) # Add the start node to the node dictionary (total cost, continuous position, (discrete parent position, continuous parent position))

            self.algorithm() # Run the algorithm
            if __name__ == "__main__": print(self.count)
            return self.interp_path

    def algorithm(self):
        timeout = time.time() + 0.16 # Exit the while loop after this time to prevent excessive lag in the simulation.
        while self.open_heap:
            if __name__ != "__main__":
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE: cfg.hybrid = not cfg.hybrid # Switch path search algorithm
                        if event.key == pygame.K_k: cfg.IDM = not cfg.IDM
                    elif event.type == pygame.QUIT: cfg.running = False # Quit the program on user intervention

                if not cfg.hybrid or not cfg.running or time.time() > timeout: break # Exit the while loop if user has switched search algorithm or terminated the program

            next(self.count)
            self.f_current, current_d = hq.heappop(self.open_heap) # Get total cost and position (x_node, y_node, theta_node) of the next node in the queue (and remove the current node)
            self.closed_dict[current_d] = self.open_dict[current_d] # Add the current node to the list of explored nodes
            self.current_d = current_d

            finished = np.all(abs(current_d[:2] - self.goal[:2]) < self.dt) & (abs(current_d[2] - self.goal[2]) < abs(self.delta[1] - self.delta[0]))
            self.success() if finished else self.iterate()

    def kinematic_model(self, current_c):
        # Current_delta stores the heading of the current node
        # Create local copies of variables
        l_f = self.l_f
        l_r = self.l_r
        dt = self.dt
        velocity = self.v
        delta = self.delta # Vector of possible steering angles

        def kinematic():
            def cell(q):
                q1 = np.around(q[0], decimals=4)
                q3 = np.around(q[1], decimals=4)
                q2 = q[2]
                return np.transpose(np.dstack((q1, q3, q2)), [2,0,1])

            def do_calc():
                beta = np.arctan((l_f / (l_f + l_r)) * np.tan(delta)) * dt # Sideslip angle
                neighbour_x_c = current_c[0] + (velocity * np.cos(current_c[2] + beta)) * dt # Next x-position (direction * cos(steer_angle + sideslip)) * timestep
                neighbour_y_c = current_c[1] + (velocity * np.sin(current_c[2] + beta)) * dt # Next y-position "" sin(steer_angle + sideslip) ""
                neighbour_theta_c = (current_c[2] + (velocity * np.sin(beta) / l_r) * dt) % (2*np.pi) # New heading angle is the current_heading_angle + (direction * sin(sideslip/b) * timestep)
                neighbours_c = np.stack((neighbour_x_c, neighbour_y_c, neighbour_theta_c))
                return neighbours_c

            neighbours_c = do_calc() # Continuous position of neighbours
            neighbours_d = cell(neighbours_c) # Discrete position of neighbours
            deltas = delta * np.ones_like(velocity) # Delta values corresponding to each neighbour
            vs = velocity * np.ones_like(delta) # Velocity values corresponding to each neighbour
            neighbours = np.concatenate((neighbours_d.T.reshape(-1,3), neighbours_c.T.reshape(-1,3), deltas.T.reshape(-1,1), vs.T.reshape(-1,1)), axis=1)

            return neighbours

        return kinematic()

    @staticmethod
    def h(position, target):
        """Heuristic function used by the algorithm. position can be an 2D numpy array."""
        h_cost = np.sum(((position - target) ** 2), axis=1 if position.ndim - 1 else None)
        return np.sqrt(h_cost)

    def iterate(self):
        f_current, current_c, _, current_v, current_delta = self.open_dict.pop(self.current_d) # Get current node data
        current_d, obstacles = self.current_d, self.obstacles

        neighbours = self.kinematic_model(current_c) # Get list of possible next positions. Each row corresponds to a neighbour. (dx, dy, dd, cx, cy, cd, d, v)

        arr, arr2 = neighbours[:, 0], neighbours[:, 1] # Preallocate for speed
        a = (arr > self.min_x) == (arr < self.max_x)
        b = (arr2 > self.min_y) == (arr2 < self.max_y)
        bounds = np.logical_and(a,b).reshape(-1,1) # Column vector. Elements are zero if corresponding neighbour is outside search area.

        def collision_check(A, B):
            """Checks if each row of A exists in B, returns 0 if True. Output is a logical vector of length num rows of A. A and B must be of size (m,k), (n,k)."""
            try: C = (A[:, np.newaxis] - B).reshape(-1, A.shape[1])  # 2 columns. num rowsC = rowsA * rowsB
            except ValueError: diff3 = np.ones((np.size(A, axis=0), 1), dtype=int) # Handles "operands could not be broadcast together with shapes (800,1,3) (0,)"
            else:
                diff = np.invert(np.array(C, dtype=bool))  # Convert C to a boolean array
                diff2 = np.logical_not(np.prod(diff, axis=1)).reshape(np.size(A, axis=0), -1)  # Each row corresponds to an item in list A. Each column corresponds to an item in list B. Element is False if all values in the corresponding row in diff are True
                diff3 = np.prod(diff2, axis=1).reshape(-1,1) # Diff3 is a logical column vector. Elements are 0 if the corresponding row in A exists in the B

            return diff3

        collision2 = np.array([obs.collidepoint(point) for obs in self.vehicle_list for point in neighbours[:, [0, 1]]]) # Check if neighbours are colliding with an obstacle
        diff = collision2.reshape(-1, np.size(neighbours[:, [0, 1]], axis=0))
        collision = ~np.any(diff, axis=0).reshape(-1, 1) # Logical vector of size (num_rows(neighbours), 1). Element is False if neighbour shares node with an obstacle.

        closed_list = [*self.closed_dict] # Get the list of nodes in the closed dictionary
        C = collision_check(neighbours[:, :3], np.array(closed_list)) # Check which neighbours are not in the closed dictionary. (Elements are True if not in the closed dictionary).
        valid_neighbours = neighbours * collision * bounds * C # Set values of invalid neighbours to 0 (ie. neighbours that have already been searched or on obstacle/out of bounds).

        def g_cost(valid_neighbours, obstacles):
            """Calculates the cost of moving from the current node to each neighbour."""
            # Obstacle cost------------
            A, B = valid_neighbours[:, 3:5], obstacles
            try: difference = (A[:, np.newaxis] - B).reshape(-1, A.shape[1])
            except ValueError: total_obs_cost = 0
            else:
                sqrdiff = difference ** 2
                sqrdiffsum = np.sum(sqrdiff, axis=1)
                proximity = np.sqrt(sqrdiffsum).reshape(np.size(A, axis=0), -1)

                prox_distance = 3 # A cost of 1 is applied if a neighbour is this many units away from an obstacle
                obstacle_cost = (prox_distance / proximity) ** 2 # Each row corresponds to a neighbour, each column corresponds to a point on an obstacle.
                total_obs_cost = np.sum(obstacle_cost, axis=1) # The total obstacle cost for each neighbour

            steer_cost = abs(valid_neighbours[:, 6] - current_delta) / self.delta.max() + 1 # Steering angle cost. The difference between the steering angle of each neighbour and the current steer angle. Normalised by the max steering angle
            reverse_cost = (valid_neighbours[:, 7] < 0) * 100 + 1 # Penalise reversing
            heading_cost = (abs(np.pi - abs((valid_neighbours[:, 5] - current_c[2]) - np.pi)) / np.pi) + 1 # Heading cost

            return total_obs_cost * steer_cost * reverse_cost * heading_cost

        g_local = g_cost(valid_neighbours, obstacles) # Cost from current node to neighbouring nodes
        g_cost = f_current - self.h(current_c[:2], self.goal[:2]) # Cost from start to the current node

        g_cost += g_local # Add the local costs to the cost so far
        f_cost = (g_cost + self.h(valid_neighbours[:, 3:5], self.goal[:2])).reshape(-1,1) # Calculate the total cost for each neighbour

        valid_neighbours_with_cost = np.concatenate((valid_neighbours, f_cost),axis=1) # Concatenate the list of costs with the modified list of neighbours.
        valid_nwc2 = valid_neighbours_with_cost[~np.all(valid_neighbours == 0,axis=1)] # Remove invalid neighbours. Last element in each row of valid2 is the cost associated with that neighbour.

        if valid_nwc2.any():  # If the current node has valid neighbours
            def sort_unique(neighbours):
                """Sorts neighbours from smallest to largest by f_cost. Only the first occurrence of each discrete position is kept."""
                sorted = neighbours[np.argsort(neighbours[:, 8])] # Sort smallest to largest by f_cost
                _, unique_index = np.unique(sorted[:, :3], return_index=True, axis=0) # Indices of unique elements with smallest f_cost
                return sorted[unique_index, :] # Return unique values with the smallest f_cost

            valid_nwc2 = sort_unique(valid_nwc2) # Sort by smallest to largest f_cost. Keep unique elements.

            open_list = list(self.open_dict.keys()) # Get the list of nodes in the open dictionary
            D = collision_check(valid_nwc2[:, :3], open_list) # Check which neighbours are not in the open dictionary. (Elements are True if not in the open dictionary).

            #Get current stored cost of items in open dictionary-------------------------
            in_open_dict = valid_nwc2 * np.logical_not(D)
            in_open_dict = valid_nwc2[~np.all(in_open_dict == 0, axis=1)] # Remove zeros

            wanted_keys = in_open_dict[:,:3]

            previous_f_costs = itemgetter(*tuple(map(tuple, wanted_keys)))(self.open_dict) if not D.all() else []
            try: previous_f_costs = np.array(list(map(itemgetter(0), previous_f_costs))) # If wanted_keys contains multiple neighbours
            except IndexError: previous_f_costs = previous_f_costs[0] # Handles "invalid index to scalar variable" (If wanted_keys only contains one neighbour)

            better = (in_open_dict[:,8] < previous_f_costs).reshape(-1, 1) # Element is True if the corresponding f_cost is lower than currently in the dictionary.

            lower_cost = in_open_dict * better # Neighbours in the open dictionary with lower cost
            lower_cost = in_open_dict[~np.all(lower_cost == 0, axis=1)] # Remove zeros

            not_in_open_dict = valid_nwc2 * D
            not_in_open_dict = valid_nwc2[~np.all(not_in_open_dict == 0, axis=1)] # Remove zeros

            update = np.concatenate((not_in_open_dict, lower_cost)) # numpy array of neighbours to update

            #Update open dictionary without for loop--------------
            rearranged = update[:, [0, 1, 2, 8, 3, 4, 5, 7, 6]]
            rearranged1, rearranged2, rearranged3, rearranged4, rearranged5 = rearranged[:,0:3], rearranged[:,3], rearranged[:,4:7], rearranged[:,7], rearranged[:,8]
            rearranged1 = tuple(map(tuple, rearranged1))  # Convert np array to tuple
            rearranged2 = tuple(rearranged2)

            new_key_values = dict(zip(rearranged1, zip(rearranged2, rearranged3, repeat((current_d, current_c)), rearranged4, rearranged5)))
            self.open_dict.update(new_key_values)
            #----------------
            #Update heapq----- Update contains items: not in open dict, in open dict with lower cost than before.
            update = update[np.argsort(update[:,8])]
            new_heap = [*zip(update[:,8], list(map(tuple, update[:,:3])))]

            [self.open_heap.remove(self.open_heap[i]) for discrete in lower_cost[:,:3] for i, v in enumerate(self.open_heap) if v[1] == tuple(discrete)] # Remove nodes from the heap that now have a lower cost

            self.open_heap.extend(new_heap)
            hq.heapify(self.open_heap)

    def success(self):
        rev_final_path = self.goal # reverse of final path
        node = self.current_d
        while np.any(~np.equal(node, self.start)):
            open_node = self.closed_dict[tuple(node)] # (total_cost, node_continuous_coords, (parent_discrete_coords, parent_continuous_coords))
            node, parent = open_node[2]
            rev_final_path = np.vstack((rev_final_path, parent))
        else:
            self.rev_final_path = rev_final_path[::-1]
            if self.rev_final_path.size > 3:
                x_path, y_path = self.rev_final_path[:,0], self.rev_final_path[:,1]
                try: tck, _ = itp.splprep([list(x_path), list(y_path)], s=0, k=3) # Draws a cubic spline between the path waypoints. s determines the amount of smoothing (0 is none). tck is a tuple containing (vector of knots, B-line coeffs, degree of the spline). u is the weighted sum of squared residuals of the spline approximation.
                except TypeError: pass
                else: self.interp_path = tck

        self.open_heap = [] # Algorithm method terminates when open_heap is empty

def plot_solution(path, obstacles):
    """Plots the solution in a new figure window.
    Path is an nx3 numpy array of x,y points. Obstacles is a list of pygame rect objects."""
    x_path, y_path = path[:, 0], path[:, 1]

    tck, u = itp.splprep([x_path, y_path], s=0, k=3) # Create the interpolated spline
    x_new, y_new = itp.splev(np.linspace(0, 1, 1000), tck) # Discretise the spline for plotting

    # Plot the path on a new figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_new, y_new, "-", color="black", label="Smoothed Path", linewidth=0.5)
    ax.plot(x_path, y_path, "|", color="black", label="Waypoints", markersize=4, linewidth=0.1)
    ax.plot(x_path[0], y_path[0], "o", color="k", label="Start", markersize=5, linewidth=0.2)
    ax.plot(x_path[-1], y_path[-1], ">", color="k", label="Goal", markersize=5, linewidth=0.2)

    rectangles = [plt.Rectangle((obs.left, obs.top), obs.width, obs.height) for obs in obstacles]
    P = PatchCollection(rectangles, fc='k')
    ax.add_collection(P)
    obs_handle = mpatches.Patch(color='black', label='Obstacles') # Manually define a new patch

    # Plot lanes:
    ax.set_yticks([*cfg.lanes.values()], minor=False)
    ax.set_yticklabels([*cfg.lanes])

    plt.xlabel("Distance")
    plt.ylabel("Lanes")

    # Draw legend
    handles, labels = ax.get_legend_handles_labels() # Get existing legend handles
    handles.append(obs_handle) # Add the obstacles patch to the list of handles
    plt.legend(handles=handles, loc='best', fancybox=False, shadow=True) # Plot the legend

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.25)
    ax.grid(which='major', alpha=0.5)

    plt.xlim([0,270])
    plt.ylim([0,120])

    ax.invert_yaxis()

    plt.show()


def main():
    # start and goal position
    # (x, y, theta) in meters, meters, degrees
    x_start, y_start, theta_start = 0, 102, 0
    x_goal, y_goal, theta_goal = 125, 89, 0

    min_x, max_x, min_y, max_y = x_start, x_goal, y_start, y_goal # Search area bounded between start and goal position
    # min_x, max_x, min_y, max_y = 0, 1530, 0, 120 # Unbounded search area


    hy_a_star = HybridAstar(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, width=1530,height=120) # Initialise Hybrid A* algorithm

    i=0
    # vehicle_list = [pygame.Rect(0 + 1.2 * i, 100, 14, 6), pygame.Rect(1454, 71, 15, 6),
    #                 pygame.Rect(200 + 1.4 * i, 72, 14, 5), pygame.Rect(250 + i, 100, 14, 5)]  # List of obstacles.
    # vehicle_list = [pygame.Rect(120 + 1.2 * i, 89, 14, 6), pygame.Rect(1454, 71, 15, 6),
    #                 pygame.Rect(200 + 1.4 * i, 72, 14, 5), pygame.Rect(250 + i, 100, 14, 5)]  # List of obstacles.
    # vehicle_list = [pygame.Rect(100, 102, 14, 6),
    #                 pygame.Rect(250, 89, 14, 5), pygame.Rect(250 + i, 102, 14, 5)]  # List of obstacles.
    # vehicle_list = []

    def getRectAround(centre_point, width, height):
        """ Return a pygame.Rect of size width by height, centred around the given centre_point """
        rectangle = pygame.Rect(0, 0, width, height)  # make new rectangle
        rectangle.center = centre_point  # centre rectangle
        return rectangle

    xs = [100, 250, 250]
    ys = [102, 89, 102]
    ws = [14, 14, 14]
    hs = [6, 5, 6]
    vehicle_list = [getRectAround((rect[0], rect[1]), rect[2], rect[3]) for rect in zip(xs, ys, ws, hs)]

    start = timer()
    path = hy_a_star.run((x_start, y_start, theta_start), (x_goal, y_goal, theta_goal), min_x, max_x, min_y, max_y, obstacles=vehicle_list) # Get the shortest path to the goal
    print(timer() - start)

    plot_solution(hy_a_star.rev_final_path, vehicle_list)

if __name__ == "__main__":
    main()