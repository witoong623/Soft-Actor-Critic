import carla

host = 'localhost'
port = 2000
map = 'ait_v4'
dt = 0.1
life_time = 120

client = carla.Client(host, port)
client.set_timeout(30.0)

client.load_world(map)
world = client.get_world()

map = world.get_map()

debug = world.debug

spawn_points = map.get_spawn_points()

for i, spawn_point in enumerate(spawn_points):
    txt = f'Spawn point {i}, X={spawn_point.location.x}, Y={spawn_point.location.y}, Z={spawn_point.location.z}'
    debug.draw_string(spawn_point.location, txt, life_time=60)
    debug.draw_point(spawn_point.location, size=0.5, life_time=life_time)
# reference_location = carla.Location(x=-75940, y=-15940)

# reference_waypoint = map.get_waypoint(reference_location)
# debug.draw_point(reference_waypoint.transform.location, size=1, life_time=life_time)

# waypoints = map.generate_waypoints(2)
# print('waypoints len', len(waypoints))
# for waypoint in waypoints:
#     location = waypoint.transform.location
#     debug.draw_point(location, size=0.5, life_time=life_time)

# location = carla.Location(x=-12220.15918, y=-2022.872314, z=0.2)

# waypoint = map.get_waypoint(location)
