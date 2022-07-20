import unittest

from unittest.mock import Mock

from common.carla_environment.route_tracker import RouteTracker


class TestManualRoutePlanner(unittest.TestCase):
    repeat_threshold = 5

    def test_cp_not_repeat_section(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), enable=False, debug_route_waypoint_len=592)
        route_planner._checkpoint_waypoint_index = 0
        route_planner._current_waypoint_index = 24

        route_planner._update_checkpoint()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)

    def test_cp_repeat_first_section(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), enable=False,
                                           repeat_section_threshold=5, debug_route_waypoint_len=592)
        route_planner._checkpoint_waypoint_index = 0
        route_planner._current_waypoint_index = 25

        route_planner._update_checkpoint()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 25)

        for i in range(1, self.repeat_threshold + 1):
            route_planner._current_waypoint_index = 25 + i
            route_planner._update_checkpoint()

            if i < self.repeat_threshold:
                self.assertEqual(route_planner._repeat_count, i + 1)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
            else:
                self.assertEqual(route_planner._checkpoint_waypoint_index, 25, f'i is {i}, count is {route_planner._repeat_count}')

        self.assertEqual(route_planner._checkpoint_waypoint_index, 25)

    def test_cp_repeat_second_section(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), enable=False, debug_route_waypoint_len=592)
        route_planner._checkpoint_waypoint_index = 25
        route_planner._current_waypoint_index = 50

        route_planner._update_checkpoint()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 25)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 50)

        # repeat same section
        for i in range(1, self.repeat_threshold+1):
            route_planner._current_waypoint_index = 50 + i
            route_planner._update_checkpoint()

            if i < self.repeat_threshold:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 25)
            else:
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0, f'i is {i}, count is {route_planner._repeat_count}')

        # repeat from the beginning
        for i in range(1, self.repeat_threshold+1):
            route_planner._current_waypoint_index = 50 + i
            route_planner._update_checkpoint()

            if i < self.repeat_threshold:
                self.assertEqual(route_planner._repeat_count, i)
                self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
            else:
                self.assertEqual(route_planner._checkpoint_waypoint_index, 50, f'i is {i}, count is {route_planner._repeat_count}')

        self.assertEqual(route_planner._checkpoint_waypoint_index, 50)


class TestSectionCheckpointUpdate(unittest.TestCase):
    repeat_threshold = 5

    def test_first_section_no_cross(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), initial_checkpoint=0,
                                           enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 35)

        route_planner._current_waypoint_index = 35
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go checkpoint 35
        for i in range(4):
            route_planner._current_waypoint_index = 35 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 35)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 70)

        # repeat 5 times to go to 0 checkpoint (because completed repeat 70 5 times)
        for i in range(5):
            route_planner._current_waypoint_index = 70 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 70)

        # repeat 5 times from the beginning to get to 70
        for i in range(5):
            route_planner._current_waypoint_index = 70 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 70)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 105)

    def test_first_section_cross(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), initial_checkpoint=105,
                                           enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 105)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 140)

        route_planner._current_waypoint_index = 142
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 105)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go 0 checkpoint
        for i in range(4):
            route_planner._current_waypoint_index = 140 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 140)

        # repeat 5 times to go to 140
        for i in range(5):
            route_planner._current_waypoint_index = 140 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 143)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 173)

    def test_second_section_cross(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), initial_checkpoint=143,
                                           enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 143)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 173)

        route_planner._current_waypoint_index = 174
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 143)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go next section checkpoint
        for i in range(4):
            route_planner._current_waypoint_index = 173 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 211)

    def test_third_section_no_cross(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), initial_checkpoint=176,
                                           enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 211)

        route_planner._current_waypoint_index = 211
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go checkpoint 211
        for i in range(4):
            route_planner._current_waypoint_index = 211 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 211)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 246)

        # ----- next section 246 -----

        # repeat 5 times to go to 176 checkpoint (because completed repeat 246 5 times)
        for i in range(5):
            route_planner._current_waypoint_index = 246 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 246)

        # repeat 5 times from the beginning to get to 246
        for i in range(5):
            route_planner._current_waypoint_index = 246
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 246)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 281)

        # ----- next section 281 -----

        # repeat 5 times to go to 176 (because complete 281 5 times)
        for i in range(5):
            route_planner._current_waypoint_index = 281 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 281)

        # repeat 5 times from the beginning to get to 281
        for i in range(5):
            route_planner._current_waypoint_index = 281 + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 281)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 316)

        # ----- next section 316 -----

        # repeat 5 times to go to 176 (because complete 316 5 times)
        for i in range(5):
            route_planner._current_waypoint_index = 316 + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 316)

        # repeat 5 times from the beginning to get to 316
        for i in range(5):
            route_planner._current_waypoint_index = 316 + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 316)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 351)

        # ----- next section 351 -----

        # repeat 5 times to go to 176 (because complete 351 5 times)
        next_wp = 351
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 351
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, next_wp)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 386)

        # ----- next section 386 -----

        # repeat 5 times to go to 176 (because complete 386 5 times)
        next_wp = 386
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 386
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, next_wp)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 421)

        # ----- next section 421 -----

        # repeat 5 times to go to 176 (because complete 421 5 times)
        next_wp = 421
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 421
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, next_wp)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 456)

        # ----- next section 456 -----

        # repeat 5 times to go to 176 (because complete 456 5 times)
        next_wp = 456
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 456
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, next_wp)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 491)

        # ----- next section 491 -----

        # repeat 5 times to go to 176 (because complete 491 5 times)
        next_wp = 491
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 491
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, next_wp)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 526)

        # ----- next section 526 -----

        # repeat 5 times to go to 176 (because complete 526 5 times)
        next_wp = 526
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 526
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, next_wp)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 561)

        # ----- next section 561 -----

        # repeat 5 times to go to 176 (because complete 561 5 times)
        next_wp = 561
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + i
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 561
        for i in range(5):
            route_planner._current_waypoint_index = next_wp + 1
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, next_wp)
        # 591 is the end of route
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 591)

        # ----- next section 0 (cross) -----

        # repeat 5 times to go to 176 (because complete 561 5 times)
        next_wp = 591
        for i in range(5):
            route_planner._current_waypoint_index = next_wp
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, next_wp)

        # repeat 5 times from the beginning to get to 561
        for i in range(5):
            route_planner._current_waypoint_index = next_wp
            route_planner._update_checkpoint_by_section()

        # cross to section 1 again
        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 35)

    def test_third_section_cross(self):
        route_planner = RouteTracker(Mock(), Mock(), world=Mock(), initial_checkpoint=561,
                                           enable=False, use_section=True, debug_route_waypoint_len=592)
        self.assertEqual(route_planner._checkpoint_waypoint_index, 561)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 0)

        route_planner._current_waypoint_index = 591
        route_planner._update_checkpoint_by_section()
        self.assertEqual(route_planner._checkpoint_waypoint_index, 561)
        self.assertEqual(route_planner._repeat_count, 1)

        # repeat same section 4 time to go 176 checkpoint
        for i in range(4):
            route_planner._current_waypoint_index = 591
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 176)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 0)

        # repeat 5 times to go to 0
        for i in range(5):
            route_planner._current_waypoint_index = 591
            route_planner._update_checkpoint_by_section()

        self.assertEqual(route_planner._checkpoint_waypoint_index, 0)
        self.assertEqual(route_planner._next_checkpoint_waypoint_index, 35)
