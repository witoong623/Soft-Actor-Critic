import unittest

from unittest.mock import Mock

from common.carla_environment.manual_route_planner import ManualRoutePlanner


class TestManualRoutePlanner(unittest.TestCase):
    def test_cp_not_repeat_section(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), enable=False)
        route_planner._checkpoint_waypoint_index = 0
        route_planner._current_waypoint_index = 24

        route_planner._update_checkpoint()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)

    def test_cp_repeat_first_section(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), enable=False)
        route_planner._checkpoint_waypoint_index = 0
        route_planner._current_waypoint_index = 25

        route_planner._update_checkpoint()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 25)

        for i in range(1, 10+1):
            route_planner._current_waypoint_index = 25 + i
            route_planner._update_checkpoint()

            if i < 10:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
            else:
                self.assertEqual(route_planner._repeat_count, 0)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 25, f'i is {i}, count is {route_planner._repeat_count}')

        self.assertEqual(route_planner._checkpoint_waypoint_index, 25)

    def test_cp_repeat_second_section(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), enable=False)
        route_planner._checkpoint_waypoint_index = 25
        route_planner._current_waypoint_index = 50

        route_planner._update_checkpoint()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 25)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 50)

        # repeat same section
        for i in range(1, 10+1):
            route_planner._current_waypoint_index = 50 + i
            route_planner._update_checkpoint()

            if i < 10:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 25)
            else:
                self.assertEqual(route_planner._repeat_count, 0)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0, f'i is {i}, count is {route_planner._repeat_count}')

        # repeat from the beginning
        for i in range(1, 10+1):
            route_planner._current_waypoint_index = 50 + i
            route_planner._update_checkpoint()

            if i < 10:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
            else:
                self.assertEqual(route_planner._repeat_count, 0)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 50, f'i is {i}, count is {route_planner._repeat_count}')

        self.assertEqual(route_planner._checkpoint_waypoint_index, 50)
