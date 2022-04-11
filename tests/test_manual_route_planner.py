import unittest

from unittest.mock import Mock

from common.carla_environment.manual_route_planner import ManualRoutePlanner


class TestManualRoutePlanner(unittest.TestCase):
    repeat_threshold = 5

    def test_cp_not_repeat_section(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), enable=False, debug_route_waypoint_len=592)
        route_planner._checkpoint_waypoint_index = 0
        route_planner._current_waypoint_index = 24

        route_planner._update_checkpoint()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)

    def test_cp_repeat_first_section(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), enable=False, debug_route_waypoint_len=592)
        route_planner._checkpoint_waypoint_index = 0
        route_planner._current_waypoint_index = 25

        route_planner._update_checkpoint()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 25)

        for i in range(1, self.repeat_threshold+1):
            route_planner._current_waypoint_index = 25 + i
            route_planner._update_checkpoint()

            if i < self.repeat_threshold:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
            else:
                self.assertEqual(route_planner._repeat_count, 0)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 25, f'i is {i}, count is {route_planner._repeat_count}')

        self.assertEqual(route_planner._checkpoint_waypoint_index, 25)

    def test_cp_repeat_second_section(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), enable=False, debug_route_waypoint_len=592)
        route_planner._checkpoint_waypoint_index = 25
        route_planner._current_waypoint_index = 50

        route_planner._update_checkpoint()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 25)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 50)

        # repeat same section
        for i in range(1, self.repeat_threshold+1):
            route_planner._current_waypoint_index = 50 + i
            route_planner._update_checkpoint()

            if i < self.repeat_threshold:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 25)
            else:
                self.assertEqual(route_planner._repeat_count, 0)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0, f'i is {i}, count is {route_planner._repeat_count}')

        # repeat from the beginning
        for i in range(1, self.repeat_threshold+1):
            route_planner._current_waypoint_index = 50 + i
            route_planner._update_checkpoint()

            if i < self.repeat_threshold:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
            else:
                self.assertEqual(route_planner._repeat_count, 0)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 50, f'i is {i}, count is {route_planner._repeat_count}')

        self.assertEqual(route_planner._checkpoint_waypoint_index, 50)


class TestSectionCheckpointUpdate(unittest.TestCase):
    repeat_threshold = 5

    def test_first_section_no_cross(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), initial_checkpoint=0, enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 35)

        route_planner._current_waypoint_index = 35
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go checkpoint 35
        for i in range(4):
            route_planner._current_waypoint_index = 35 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 35)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 70)

        # repeat 5 times to go to 0 checkpoint (because completed repeat 70 5 times)
        for i in range(5):
            route_planner._current_waypoint_index = 70 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 70)

        # repeat 5 times from the beginning to get to 70
        for i in range(5):
            route_planner._current_waypoint_index = 70 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 70)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 105)

    def test_first_section_cross(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), initial_checkpoint=105, enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 105)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 140)

        route_planner._current_waypoint_index = 140
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 105)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go 0 checkpoint
        for i in range(4):
            route_planner._current_waypoint_index = 140
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 140)

        # repeat 5 times to go to 140
        for i in range(5):
            route_planner._current_waypoint_index = 140
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 143)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 173)

    def test_second_section_cross(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), initial_checkpoint=143, enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 143)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 173)

        route_planner._current_waypoint_index = 173
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 143)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go next section checkpoint
        for i in range(4):
            route_planner._current_waypoint_index = 173
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 211)

    def test_third_section_no_cross(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), initial_checkpoint=176, enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 211)

        route_planner._current_waypoint_index = 211
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go checkpoint 211
        for i in range(4):
            route_planner._current_waypoint_index = 211
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 211)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 246)

        # repeat 5 times to go to 176 checkpoint (because completed repeat 246 5 times)
        for i in range(5):
            route_planner._current_waypoint_index = 246
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 246)

        # repeat 5 times from the beginning to get to 246
        for i in range(5):
            route_planner._current_waypoint_index = 246
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 246)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 281)

    def test_third_section_cross(self):
        route_planner = ManualRoutePlanner(Mock(), Mock(), initial_checkpoint=561, enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 561)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 0)

        route_planner._current_waypoint_index = 591
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 561)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go 176 checkpoint
        for i in range(4):
            route_planner._current_waypoint_index = 591
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 0)

        # repeat 5 times to go to 0
        for i in range(5):
            route_planner._current_waypoint_index = 591
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._intermediate_checkpoint_waypoint_index, 35)
