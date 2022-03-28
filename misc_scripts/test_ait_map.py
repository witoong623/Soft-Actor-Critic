import carla

host = 'localhost'
port = 2000

client = carla.Client(host, port)
print(client.get_available_maps())
# world = client.load_world('ait_v4')

# spawn_points = world.get_map().get_spawn_points()
# print(f'there are {len(spawn_points)} spawn points')
