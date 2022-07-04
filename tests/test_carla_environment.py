import unittest
import cv2
import random
import numpy as np
import numba
import time
import carla

from numba.np.extensions import cross2d
from numba.typed import List
from common.carla_environment.misc import get_lane_dis
from common.carla_environment.manual_route_planner import ManualRoutePlanner, TOWN7_PLAN
from common.carla_environment.misc import is_the_same_direction


class TestObservationTransform(unittest.TestCase):
    W = 512
    H = 256

    def test_rgb_bgr_not_equal(self):
        bgr_img = cv2.imread('images/architecture.png')
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(bgr_img, rgb_img)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(bgr_img, bgr_img[:,:,::-1])

    def test_resize_INTER_NEAREST(self):
        bgr_img = cv2.imread('images/architecture.png')
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        bgr_resized_img = cv2.resize(bgr_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        rgb_resized_img = cv2.resize(rgb_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        np.testing.assert_array_equal(bgr_resized_img, rgb_resized_img[:,:,::-1])
        np.testing.assert_array_equal(bgr_resized_img[:,:,::-1], rgb_resized_img)

    def test_resize_default(self):
        bgr_img = cv2.imread('images/architecture.png')
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        bgr_resized_img = cv2.resize(bgr_img, (self.W, self.H))
        rgb_resized_img = cv2.resize(rgb_img, (self.W, self.H))

        np.testing.assert_array_equal(bgr_resized_img, rgb_resized_img[:,:,::-1])
        np.testing.assert_array_equal(bgr_resized_img[:,:,::-1], rgb_resized_img)


@numba.jit(nopython=True, cache=True)
def get_lane_dis_numba(waypoints, x, y):
    """
    Calculate distance from (x, y) to waypoints.
    :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
    :param x: x position of vehicle
    :param y: y position of vehicle
    :return: a tuple of the distance and the closest waypoint orientation
    """
    dis_min = 1000
    waypt = waypoints[0]
    for pt in waypoints:
        # distance between two 2D vectors
        d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
        if d < dis_min:
            dis_min = d
            waypt = pt

    vec = np.array((x - waypt[0], y - waypt[1]), dtype=np.float32)
    lv = np.linalg.norm(vec)
    w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
    cross = cross2d(w, vec/lv)
    dis = - lv * cross
    return dis, w


class TestNumbaCalculation(unittest.TestCase):
    def test_distance_calculation(self):
        x = 1
        y = 1

        waypoints_x = [random.randrange(4, 100) for i in range(200)]
        waypoints_y = [random.randrange(4, 100) for i in range(200)]
        waypoints_yaw = [random.randrange(4, 100) for i in range(200)]
        waypoints = zip(waypoints_x, waypoints_y, waypoints_yaw)
        waypoints = [(wp[0], wp[1], wp[2]) for wp in waypoints]
        waypoints_numba = List(waypoints)

        # warm up cache loading
        dis_numba, w_numba = get_lane_dis_numba(waypoints_numba, x, y)

        t = time.process_time_ns()
        dis_numba, w_numba = get_lane_dis_numba(waypoints_numba, x, y)
        elapsed = time.process_time_ns() - t
        print('numba time', elapsed)

        t = time.process_time_ns()
        dis, w = get_lane_dis(waypoints_numba, x, y)
        elapsed = time.process_time_ns() - t
        print('pure python time', elapsed)

        # dis_numba is float, dis is np.float64
        # self.assertEqual(type(dis_numba), type(dis))
        self.assertEqual(type(w_numba), type(w))

        self.assertAlmostEqual(dis_numba, dis, places=6)
        np.testing.assert_array_equal(w_numba, w)


class TestDirectionCalculation(unittest.TestCase):
    def setUp(self) -> None:
        host = 'localhost'
        port = 2000
        map = 'Town07'

        self.client = carla.Client(host, port)

        self.world = self.client.load_world(map)

        vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        lap_spwan_point_wp = self.world.get_map().get_waypoint(vehicle_spawn_points[1].location)

        self.routeplanner = ManualRoutePlanner(lap_spwan_point_wp, lap_spwan_point_wp, world=self.world, resolution=2, plan=TOWN7_PLAN)

    def test_waypoint_direction_backward(self):
        ''' If car ran passed `current waypoint`,
            what projection from current location of car will look like? '''
        waypoints = self.routeplanner.get_route_waypoints()

        start = 130
        end = 150
        for i, wps in enumerate(waypoints[start:end], start):
            wp, _ = wps

            for j in range(start-1, start-5, -1):
                # current_wp is used inplace of current car location
                current_wp = wp
                # next_wp is the waypoint that car suppose to move to but it already passed
                next_wp, action = waypoints[j]
                is_same_direction = is_the_same_direction(action, current_transform=current_wp.transform, waypoint_transform=next_wp.transform)
                self.assertTrue(is_same_direction)
