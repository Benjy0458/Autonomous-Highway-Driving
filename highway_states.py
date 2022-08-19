# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:52:31 2022
V5: In this version, the left and rights have been swapped for driving on the right.
V6: The on_event method now takes 2 arguments, observation and the current lane of the agent. Now the Right Lane Change
State is unavailable if the agent is in the right lane etc.
@author: benjy
"""

from state import State

class FreeRideState(State):
    """The state which indicates that there are no other vehicles in front of the AV."""
    def on_event(self, event, lane):
        if event == 'right_lane_free':
            if lane == 5: pass
            else: return LaneChangeRightState()
        elif event == 'clear_road': pass
        elif event == 'vehicle_ahead' or 'slow_vehicle': return FollowState()

        return self

class FollowState(State):
    """The state which indicates that the AV is following a lead vehicle."""
    def on_event(self, event, lane):
        if event == 'clear_road': return FreeRideState()
        elif event == 'slow_vehicle':
            if lane == 3: pass
            else: return LaneChangeLeftState()
        elif event == 'right_lane_free': return LaneChangeRightState()

        return self

class LaneChangeLeftState(State):
    """The state which indicates a left lane change."""
    def on_event(self, event, lane):
        if event == 'slow_vehicle':
            if lane == 3: return FollowState()
            else: pass
        elif event == 'clear_road': return FreeRideState()
        elif event == 'vehicle_ahead' or 'slow_vehicle': return FollowState()

        return self

class LaneChangeRightState(State):
    """The state which indicates a right lane change."""
    def on_event(self, event, lane):
        if event == 'right_lane_free':
            if lane == 5: return FreeRideState()
            else: pass
        elif event == 'vehicle_ahead' or 'slow_vehicle': return FollowState()
        elif event == 'clear_road': return FreeRideState()

        return self
