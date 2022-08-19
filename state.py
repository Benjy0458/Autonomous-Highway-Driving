# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:17:42 2022

@author: benjy
"""

class State(object):
    """Defines a state object which provides some utility functions for the individual states within the state machine.
    """
    def __init__(self):
        pass

    def on_event(self, event):
        """Handle events that are delegated to this state."""
        pass

    def __repr__(self):
        """Leverages the __str__ method to describe the state."""
        return self.__str__()

    def __str__(self):
        """Returns the name of the state."""
        return self.__class__.__name__