import sys
sys.path.append('/home/witoon/thesis/code/Soft-Actor-Critic')

import carla
import matplotlib.pyplot as plt

from common.carla_environment.manual_route_planner import ManualRoutePlanner, TOWN7_PLAN



host = 'localhost'
port = 2000
map = 'Town07'
dt = 0.1

client = carla.Client(host, port)
client.set_timeout(30.0)

world = client.load_world(map)

# settings = world.get_settings()
# settings.fixed_delta_seconds = dt
# settings.synchronous_mode = True
# world.apply_settings(settings)

vehicle_spawn_points = list(world.get_map().get_spawn_points())

z_locations = [t.location.z for t in vehicle_spawn_points]

plt.plot(z_locations)
plt.savefig('z_spawnpoint.jpg')
