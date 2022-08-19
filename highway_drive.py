# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:54:45 2022

@author: benjy
"""

from highway_states import FreeRideState

class HighwayDrive(object):
    """A simple state machine that mimics basic highway driving behaviours."""
    def __init__(self):
        """ Initialise the components. """
        self.state = FreeRideState() # Default state.

    def on_event(self, event, lane):
        """Incoming events are delegated to the given state which then handles the event.
        The result is assigned as the new state."""
        self.state = self.state.on_event(event, lane) # The next state will be the result of the on_event function.