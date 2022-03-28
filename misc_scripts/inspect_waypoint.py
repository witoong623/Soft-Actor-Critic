import carla

from common.carla_environment.manual_route_planner import ManualRoutePlanner, TOWN4_PLAN



host = 'localhost'
port = 2000
map = 'Town07'
dt = 0.1

client = carla.Client(host, port)
client.set_timeout(30.0)

world = client.load_world(map)

settings = world.get_settings()
# settings.fixed_delta_seconds = dt
# settings.synchronous_mode = True
# world.apply_settings(settings)

vehicle_spawn_points = list(world.get_map().get_spawn_points())
lap_spwan_point_wp = world.get_map().get_waypoint(vehicle_spawn_points[1].location)

routeplanner = ManualRoutePlanner(lap_spwan_point_wp, lap_spwan_point_wp, resolution=5, plan=TOWN4_PLAN)

bp_library = world.get_blueprint_library()

ego_bp = bp_library.find('vehicle.nissan.micra')

ego_spwan_transform = routeplanner.spawn_transform
ego_vehicle = world.try_spawn_actor(ego_bp, ego_spwan_transform)

routeplanner.set_vehicle(ego_vehicle)
waypoints = routeplanner.run_step()

print(f'waypoints len is {len(waypoints)}')

debug = world.debug
green = carla.Color(r=0, g=255, b=0)
for wp in waypoints:
    location = carla.Location(x=wp[0], y=wp[1], z=1.0)
    debug.draw_point(location, size=0.3, life_time=60, color=green)

world.wait_for_tick()