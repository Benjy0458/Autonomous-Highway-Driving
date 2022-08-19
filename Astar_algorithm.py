"""
V4:
- Global variables imported using cfg module. Eg. running state of the program.
- Algorithm while loop will terminate early if the user presses the space bar.


f(n) = g(n) + h(n)
n is the next node on the path.
g(n) is the cost of the path from the start node to n
h(n) is a heuristic function that estimates the cost of the cheapest path from n to the goal.
Heuristic function should never overestimate the actual cost to get to the goal.

Implementation:
Use a priority queue (open set or fringe) to perform repeated selection of minimum cost nodes to expand.
At each step:
    Remove the node with the the lowest f(x) value.
    If the node is the goal node, Finish
    Update the f and g values of its neighbours.
    Add the neighbours to the queue.
    Keep track of the current sequence of nodes.


Inside loop:
inputs: Current position, Goal position, obstacle locations


"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as itp
import random
import time
import pygame
from queue import PriorityQueue
import inspect
import sys
from threading import Thread
from multiprocessing import Process

from itertools import count

import cfg

def init_pygame(WIDTH, HEIGHT):
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))  # Opens a new pygame window
    pygame.display.set_caption("A* Algorithm")  # Sets the caption
    return WINDOW

class Node:
    """Nodes for A* algorithm."""
    colours = {
        "RED": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "BLUE": (0, 0, 255),
        "YELLOW": (255, 255, 0),
        "WHITE": (255, 255, 255),
        "BLACK": (0, 0, 0),
        "PURPLE": (128, 0, 128),
        "ORANGE": (255, 165, 0),
        "GREY": (128, 128, 128),
        "SILVER": (192, 192, 192),
        "GAINSBORO": (220, 220, 220),
        "TURQUOISE": (64, 224, 208)
    }

    def __init__(self, row, col, width, height, total_rows, total_cols):
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.x = col * width  # x-coordinate of node converted from row position (increases right)
        self.y = row * height  # y-coordinate of node converted from column position (increases down)
        self.colour = self.colours["WHITE"]
        self.neighbours = []

    def get_pos(self):
        return self.row, self.col  # Determines row and column position of a particular node

    def is_closed(self):
        return self.colour == self.colours["RED"]  # Visited nodes = Red

    def is_open(self):
        return self.colour == self.colours["GREEN"]  # Nodes to be visited = Green

    def is_obstacle(self):
        return self.colour == self.colours["BLACK"]  # Obstacles = Black

    def is_1_from_obstacle(self):
        return self.colour == self.colours["SILVER"]  # Nodes 1 away from obstacle = Silver

    def is_2_from_obstacle(self):
        return self.colour == self.colours["GAINSBORO"]  # Nodes 2 away from obstacle = GAINSBORO

    def is_start(self):
        return self.colour == self.colours["ORANGE"]  # Start node = Orange

    def is_goal(self):
        return self.colour == self.colours["TURQUOISE"]  # Goal node = Turquoise

    def reset(self):
        self.colour = self.colours["WHITE"]  # Colour all reset nodes white

    def make_closed(self):
        self.colour = self.colours["RED"]  # Colour all visited nodes red

    def make_open(self):
        self.colour = self.colours["GREEN"]  # Colour all nodes to be visited green

    def make_obstacle(self):
        self.colour = self.colours["BLACK"]  # Colour all non-traversable nodes (i.e. obstacles) black

    def make_1_from_obstacle(self):
        self.colour = self.colours["SILVER"]  # Colour all nodes 1 away from obstacle silver

    def make_2_from_obstacle(self):
        self.colour = self.colours["GAINSBORO"]  # Colour all nodes 2 away from obstacle gainsboro

    def make_start(self):
        self.colour = self.colours["ORANGE"]  # Colour start node orange

    def make_goal(self):
        self.colour = self.colours["TURQUOISE"]  # Colour goal node turquoise

    def make_path(self):
        self.colour = self.colours["PURPLE"]  # Colour nodes of shortest path purple

    def draw(self, window):
        pygame.draw.rect(window, self.colour, (self.x, self.y, self.width, self.height))

    def update_neighbours(self, grid):
        """Considers movements to any of the 8 surrounding nodes."""
        neighbours = []
        row = self.row
        col = self.col
        total_rows = self.total_rows
        total_cols = self.total_cols
        append = neighbours.append

        # Criteria for moving DOWN 1 row
        # If no obstacle in node below current node and bottom row not yet reached, move down 1 row
        if row < total_rows - 1 and not grid[row + 1][col].is_obstacle(): append(grid[row + 1][col])

        # Criteria for moving UP 1 row
        # If no obstacle in node above current node and top row not yet reached, move up 1 row
        if row > 0 and not grid[row - 1][col].is_obstacle(): append(grid[row - 1][col])

        # Criteria for moving RIGHT 1 column
        # If no obstacle in node to right of current node and rightmost column not yet reached, move right 1 column
        if col < total_cols - 1 and not grid[row][col + 1].is_obstacle(): append(grid[row][col + 1])

        # Criteria for moving LEFT 1 column
        # If no obstacle in node to left of current node and leftmost column not yet reached, move left 1 column
        if col > 0 and not grid[row][col - 1].is_obstacle(): append(grid[row][col - 1])

        # Criteria for moving diagonally
        if row > 0 and not grid[row - 1][col].is_obstacle():
            if (col > 0 and not grid[row][col - 1].is_obstacle()) and not grid[row - 1][col - 1].is_obstacle(): append(grid[row - 1][col - 1]) # UP LEFT
            if (col < total_cols - 1 and not grid[row][col + 1].is_obstacle()) and not grid[row - 1][col + 1].is_obstacle(): append(grid[row - 1][col + 1]) # UP RIGHT

        if row < total_rows - 1 and not grid[row + 1][col].is_obstacle():
            if (col > 0 and not grid[row][col - 1].is_obstacle()) and not grid[row + 1][col - 1].is_obstacle(): append(grid[row + 1][col - 1]) # DOWN LEFT
            if (col < total_cols - 1 and not grid[row][col + 1].is_obstacle()) and not grid[row + 1][col + 1].is_obstacle(): append(grid[row + 1][col + 1]) # DOWN RIGHT

        self.neighbours = neighbours

    def __lt__(self, other):
        return False

class Astar:
    Draw = False # Draw the nodes on the pygame window

    def __init__(self, WIDTH, HEIGHT, rows, cols, window = None):
        self.window = window # Pygame display surface
        self.rows, self.cols = rows, cols # Grid density, number of rows and columns
        self.width, self.height = WIDTH, HEIGHT  # Width, Height of the pygame window
        self.row_gap = HEIGHT // rows  # Distance between rows (pixels)
        self.col_gap = WIDTH // cols  # Distance between cols (pixels)
        self.grid = self.make_grid() # Create the grid of nodes
        self.reset()

    def reset(self):
        grid = self.grid
        [grid[i][j].reset() for i in range(self.rows + 1) for j in range(self.cols + 1)] # Reset the colour of each node

        self.interp_path = () # Stores the tck tuple containing the interpolated shortest path.
        self.path_pos, self.x_path, self.y_path, = [], [], [] # Used in path(path_pos, x_path, y_path), algorithm and main functions

        self.g = {} # Dictionary to store the cost of each node.
        self.f = self.g.copy() # Set the total cost of each node to infinity.

        self.obs_xy = ([], []) # Lists of ([row], [col]) values of obstacles
        self.count = count() # Number of nodes explored
        self.open_set = PriorityQueue() # Queue of nodes to be explored next. Entries are kept sorted by heapq module. Lowest valued entry is returned next.
        self.open_set_hash = {} # Keeps track of nodes in the priority queue (Same information as self.open_set)
        self.came_from = {} # Dictionary of preceding nodes
        self.current = 0 # Stores the current node

        if __name__ == "__main__":
            image = pygame.image.load(r'11_highway.jpg')
            image = pygame.transform.scale(image, (self.width, 118))
            self.window.blit(image, (0, 0))  # Fill window with the highway image

    def make_grid(self):
        """Creates a grid of nodes"""
        grid = [[Node(i, j, self.col_gap, self.row_gap, self.rows, self.cols) for j in range(self.cols + 1)] for i in range(self.rows + 1)]
        return grid

    def transform_coord(self, pos):
        """Converts a position in pixels to (row, col)"""
        x, y = pos
        row = y // self.row_gap
        col = x // self.col_gap
        return int(row), int(col)

    def make_node(self, pos):
        """Accepts a raw (x, y) position and returns the corresponding node and its primary and secondary neighbours"""
        row, col = self.transform_coord(pos)
        try: node = self.grid[row][col]  # Finds row and column of node
        except IndexError: pass
        else:
            neighbours, neighbours_2 = [], []
            if row > 0: neighbours.append(self.grid[row - 1][col])
            if row < self.rows - 1: neighbours.append(self.grid[row + 1][col])
            if col > 0: neighbours.append(self.grid[row][col - 1])
            if col < self.cols - 1: neighbours.append(self.grid[row][col + 1])

            if row > 1: neighbours_2.append(self.grid[row - 2][col])
            if row < self.rows - 2: neighbours_2.append(self.grid[row + 2][col])
            if col > 1: neighbours_2.append(self.grid[row][col - 2])
            if col < self.cols - 2: neighbours_2.append(self.grid[row][col + 2])

            if row > 0 and col > 0: neighbours.append(self.grid[row - 1][col - 1])
            if row < self.rows - 1 and col < self.cols - 1: neighbours.append(self.grid[row + 1][col + 1])
            if row > 0 and col < self.cols - 1: neighbours.append(self.grid[row - 1][col + 1])
            if row < self.rows - 1 and col > 0: neighbours.append(self.grid[row + 1][col - 1])

            return node, neighbours, neighbours_2

    def create_obstacles(self, vehicle_list):
        """Accepts a list of pygame rectangle objects. Each point on the rectangle is converted to an obstacle node.
        points = ['topleft', 'bottomleft', 'topright', 'bottomright', 'midtop', 'midleft', 'midbottom', 'midright', 'center']"""
        x_obs, y_obs = [], []  # Empty lists to store obstacle x,y values
        points = {23, 24, 25, 31, 32, 33, 34, 38, 39}  # Index of inbuilt points in pygame.rect
        locations = (inspect.getmembers(vehicle, lambda a: not (inspect.isroutine(a))) for vehicle in vehicle_list) # Get the list of attributes associated with the vehicle rects.
        points_list = (veh[point][1] for veh in locations for point in points) # Get all the discrete points of the pygame rects.
        for point in points_list: # Update neighbours of each obstacle point
            try: node, neighbours, neighbours_2 = self.make_node(point) # Create obstacle node
            except TypeError: continue # Handle TypeError: cannot unpack non-iterable NoneType object
            else:
                if node.get_pos() == self.goal.get_pos(): return None
                else:
                    [neighbour.make_1_from_obstacle() for neighbour in neighbours if not neighbour.is_obstacle()] # If immediate neighbour is not an obstacle, make it dark grey
                    [neighbour_2.make_2_from_obstacle() for neighbour_2 in neighbours_2 if not neighbour_2.is_obstacle() and not neighbour_2.is_1_from_obstacle()] # If the secondary neighbour isn't - or isn't next to - an obstacle, make it light grey
                    node.make_obstacle() # Change colour to black

                    row, col = self.transform_coord(point)
                    x, y = [col + 1 / 2, ((self.rows - 1) - row) + 1 / 2]  # (x, y) position of the obstacle
                    x_obs.append(x), y_obs.append(y) # Update the list of obstacle x values

        return x_obs, y_obs

    def draw(self):
        """Draws the grid in the pygame window"""
        if __name__ == "__main__":
            image = pygame.image.load(r'11_highway.jpg')
            image = pygame.transform.scale(image, (self.width, 118))
            self.window.blit(image, (0, 0)) # Fill window with the highway image

        [node.draw(self.window) for row in self.grid for node in row if node.colour not in ([node.colours["WHITE"]] if self.Draw else (node.colours["WHITE"], node.colours["RED"], node.colours["GREEN"]))] # Colour in all the nodes [node.colours["WHITE"]]: # node.colours["BLACK"], node.colours["GREY"], node.colours["GAINSBORO"], node.colours["SILVER"]
        __name__ == "__main__" and self.Draw and self.draw_grid()
        pygame.display.update() and  __name__ == "__main__"

    def draw_grid(self):
        """Draws gridlines on grid"""
        [pygame.draw.line(self.window, (128, 128, 128), (0, i * self.row_gap), (self.width, i * self.row_gap)) for i in range(self.rows)] # Draws horizontal gridlines
        [pygame.draw.line(self.window, (128, 128, 128), (j * self.col_gap, 0), (j * self.col_gap, self.height)) for j in range(self.cols)] # Draws vertical gridlines

    def run(self, start_pos, goal_pos, obstacles):
        """Runs the A* algorithm. obstacles is a list of pygame rect objects."""
        a = goal_pos #(x, y)
        b = (self.width, self.height) # (x, y)
        if all(ai < bi for ai,bi in zip(a,b)):
            self.reset() # Reset the state of all nodes and their corresponding costs.
            self.start_pos, self.goal_pos = start_pos, goal_pos # x,y start and goal position.
            try:
                self.start, _, _ = self.make_node(start_pos)  # Start node
                self.goal, _, _ = self.make_node(goal_pos)  # Goal node
            except TypeError: pass
            else:
                self.x_goal = self.goal.get_pos()[1] + 1 / 2
                self.y_goal = self.goal.get_pos()[0] + 1 / 2
                self.obs_xy = self.create_obstacles(obstacles) # List of (x, y) values of obstacles
                if self.obs_xy:
                    self.start.make_start(), self.goal.make_goal()  # Change node colours

                    self.g[self.start] = 0 # Set the cost of the start node to 0
                    self.f[self.start] = self.h(self.start.get_pos(), self.goal.get_pos()) # Total cost at start node is just the heuristic cost.

                    self.open_set.put((self.f[self.start], next(self.count), self.start)) # Add the start node to the priority queue. will be explored first (total cost, node_number, node)
                    self.open_set_hash = {self.start} # Keeps track of nodes in the priority queue (Same information as self.open_set)

                    # __name__ == "__main__" and self.draw and self.draw() # Draws grid on window
                    self.algorithm()  # Run the algorithm
                    return self.interp_path

    def algorithm(self):
        # While the priority queue isn't empty
        while not self.open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: cfg.hybrid = not cfg.hybrid  # Switch path search algorithm
                    if event.key == pygame.K_k: cfg.IDM = not cfg.IDM # Switch controller type
                elif event.type == pygame.QUIT and __name__ != "__main__": cfg.running = False  # Quit the program on user intervention

            if __name__ != "__main__" and (cfg.hybrid or not cfg.running): break  # Exit the while loop if user has switched search algorithm or terminated the program.

            self.current = self.open_set.get()[2] # Get the next node in the priority queue (the node was stored at index 2)
            self.current.update_neighbours(self.grid) # Update the neighbours of the current node
            self.open_set_hash.remove(self.current) # Remove current node from open set
            self.success() if self.current == self.goal else self.iterate() # If we haven't found the goal node, iterate through each neighbour of the current node.
            self.current != self.start and self.current.make_closed() # If current node is not start node, colour it red

    def h(self, p1, p2):
        """Heuristic defined as Euclidean distance between goal and current nodes."""
        x1, y1 = p1
        x2, y2 = p2
        # Manhattan distance = Distance in x-direction + Distance in y-direction
        # return abs(x2 - x1) + abs(y2 - y1)

        # Octile heuristic
        x, y = abs(x2 - x1), abs(y2 - y1)*2
        e = 1 / (self.cols ** 5.8)  # Breaks ties between nodes with same f-value by favouring nodes closer to goal.
        hO = (max(x, y) + (np.sqrt(2) - 1) * min(x, y)) * (1 + e)
        # return (max(x, y) + (np.sqrt(2) - 1) * min(x, y)) * (1 + e)

        # Heuristic = Euclidean distance * f-score tie-breaker factor p
        e = 1 / (self.cols ** 2)  # Breaks ties between nodes with same f-value by favouring nodes closer to goal
        hE = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * (1 + e)
        # return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * (1 + e)

        return (hO)

    def iterate(self):
        """Iterates through each neighbour of the current node"""
        # Iterate through each neighbour of the current node
        current = self.current
        for neighbour in current.neighbours:
            # Apply a cost if a neighbouring node is near an obstacle.
            if neighbour.is_1_from_obstacle(): cost = 10
            elif neighbour.is_2_from_obstacle(): cost = 5
            else: cost = 1

            temp_g = self.g[current] + cost # Update the provisional cost of the current node.

            try: self.g[neighbour]
            except KeyError: self.g[neighbour] = float('inf') # If the neighbour hasn't been explored set its cost to inf.

            """If the g cost of the current node is less than the cost of the neighbour, the neighbour comes from the current node.
            The total cost of the neighbour is its g cost plus the h cost."""
            # If the cost of the current node is less than the cost of the neighbouring node.
            if temp_g < self.g[neighbour]:  # Update cost of node so that it is minimum possible value
                self.came_from[neighbour] = current  # Neighbouring node comes from current node
                self.g[neighbour] = temp_g
                self.f[neighbour] = temp_g + self.h(neighbour.get_pos(), self.goal.get_pos())

                """Add the neighbour to the open list if it hasn't been searched."""
                if neighbour not in self.open_set_hash and not neighbour.is_closed():
                    self.open_set.put((self.f[neighbour], next(self.count), neighbour))  # Add the node to the priority queue
                    self.open_set_hash.add(neighbour)
                    neighbour.make_open()  # Change node colour to green

    def success(self):
        while not self.open_set.empty(): # Empty the priority queue.
            try: self.open_set.get(False)
            except Empty: continue
            self.open_set.task_done()

        self.path_pos.append((self.x_goal, self.y_goal))  # Add the (x, y) position of the goal to the list of waypoints. Probably inaccurate since determined from low res (row, col) position
        self.path()  # Draw shortest path by colouring all relevant nodes purple
        self.start.make_start(), self.goal.make_goal() # Ensures start and goal nodes retain original colour when drawing path
        (__name__ == "__main__") and self.draw() # Draws grid on window

        try:
            x_path, y_path = [x * self.col_gap for x in self.x_path], [y * self.row_gap for y in self.y_path] # Convert back to pygame window coords.
            tck, _ = itp.splprep([x_path, y_path], s=0, k=3)  # Draws a cubic spline between the path waypoints. s determines the amount of smoothing (0 is none). tck is a tuple containing (vector of knots, B-line coeffs, degree of the spline). u is the weighted sum of squared residuals of the spline approximation.
        except TypeError: pass
        else:
            self.interp_path = tck
            if __name__ == "__main__":
                xs, ys = itp.splev(np.linspace(0, 1, 100), tck)  # Discrete coordinates for plotting the spline
                [pygame.draw.line(self.window, (0, 250, 250), (xs[i], ys[i]), (xs[i+1], ys[i+1])) for i in range(len(xs)-1)]
                pygame.display.update()

        # self.plot_solution(xs / self.col_gap, ys / self.row_gap) # Plot the solution in a new figure window.

    def path(self):
        """Sketches shortest path from goal node to start node."""
        while self.current in self.came_from:  # Keep drawing path until all nodes in "came_from" are coloured purple
            self.current = self.came_from[self.current]  # Back-propagate through path nodes in shortest path. Path only includes nodes in "came_from"
            y, x = [n + 0.5 for n in self.current.get_pos()]
            self.current.make_path() # Sets purple colour
            self.path_pos.append((x, y)) # Add the current node position to the list of path waypoints.
            # __name__ == "__main__" and self.draw()  # Draws grid on window

        [(self.x_path.append(insta_pos[0]), self.y_path.append(insta_pos[1])) for insta_pos in self.path_pos[::-1]]

    def plot_solution(self, x_new, y_new):
        """Plots the solution in a new figure window."""
        fig, ax = plt.subplots(figsize=(13, 3))
        ax.plot(self.x_path, self.y_path, "-", color="darkgreen", label="Shortest Path")
        ax.plot(x_new, y_new, "-", color="lime", label="Smoothed Path")
        ax.plot(self.x_path, self.y_path, "x", color="purple", label="Waypoints")
        ax.plot(self.x_path[0], self.y_path[0], "x", color="orange", label="Start Node")
        ax.plot(self.x_path[-1], self.y_path[-1], "x", color="mediumturquoise",
                label="Goal Node")
        ax.plot(self.obs_xy[0], [self.rows - y for y in self.obs_xy[1]], "s", color="k", label="Obstacles")
        ax.text(self.x_path[0], self.y_path[0], "S", fontsize=12)
        ax.text(self.x_path[-1], self.y_path[-1], "G", fontsize=12)
        plt.title("A* Shortest Path in Cartesian Coordinate System")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.legend(loc='best', fancybox=True, shadow=True)

        major_ticks_y = np.arange(0, self.rows + 1, 1 * 5)
        minor_ticks_y = np.arange(0, self.rows + 1, 1)
        major_ticks_x = np.arange(0, self.cols + 1, 1 * 5)
        minor_ticks_x = np.arange(0, self.cols + 1, 1)

        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)

        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(minor_ticks_y, minor=True)

        ax.invert_yaxis()

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.25)
        ax.grid(which='major', alpha=0.5)

        plt.show()

#---------------------------------
def main():
    def prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    WIDTH, HEIGHT = 1530, 120  # Width, height of Pygame window
    print(prime_factors(WIDTH))
    rows, cols = 60, 170 # rows, cols should be factors of HEIGHT, WIDTH respectively. Sets the grid density, Number of rows/columns in grid

    WINDOW = init_pygame(WIDTH, HEIGHT) # Create the pygame window
    A_star = Astar(WIDTH, HEIGHT, rows, cols, WINDOW) # Initialise the Astar algorithm
    t = []
    for i in range(0, 1000, 10):
        agent_pos, goal_pos = (0+i, 100), (250+i, 89)  # Start and end position of the algorithm.
        # vehicle_list = [pygame.Rect(120+1.2*i, 89, 14, 6), pygame.Rect(1454, 71, 15, 6), pygame.Rect(200+1.4*i, 72, 14, 5),
        #                 pygame.Rect(250+i, 100, 14, 5)]  # List of obstacles.
        vehicle_list = []
        start = time.time()
        path = A_star.run(agent_pos, goal_pos, vehicle_list)  # Run the algorithm
        end = time.time()
        t.append(end - start)
        print(end - start)

    print()
    print(sum(t) / len(t))

    # Ensures Pygame window stays open until closed by user.
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()
