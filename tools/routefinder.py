import carla
import curses
from blessed import Terminal


class TravelManager:
    def __init__(self, world, start_waypoint, distance):
        self.world = world
        self.debug = world.debug

        self.distance = distance
        self.start_waypoint = start_waypoint
        self.current_waypoint = start_waypoint
        self.previous_waypoint = None
        self.choices = [start_waypoint]

        self.func_call_history = []

        self._waypoint_equal_count = 1

        self._red = carla.Color(r=255, g=0, b=0)
        self._green = carla.Color(r=0, g=255, b=0)
        self._blue = carla.Color(r=0, g=0, b=255)
        self._white = carla.Color(r=255, g=255, b=255)
        self._waypoint_life_time = 30

    def call_next(self):
        self.choices = self.current_waypoint.next(self.distance)
        self.func_call_history.append('next')

    def call_previous(self):
        self.choices = self.current_waypoint.previous(self.distance)
        self.func_call_history.append('previous')

    def increase_distance_when_have_choices(self):
        ''' So that it chooses correct path at junction '''
        previous_get_waypoint_func = self.func_call_history[-1]
        wp_choices = self.choices
        step = self.distance
        while len(wp_choices) > 1:
            if previous_get_waypoint_func == 'next':
                wp_choices = self.current_waypoint.next(step)
            else:
                wp_choices = self.current_waypoint.previous(step)

            wp0, wp1 = wp_choices[:2]
            if wp0.transform.location.distance(wp1.transform.location) < self.distance:
                step += self.distance
            else:
                self.choices = wp_choices
                break

    def revert(self):
        pass

    def have_choices(self):
        return len(self.choices) > 1

    def have_one_choice(self):
        return len(self.choices) == 1

    def select_waypoint(self, choice_index):
        self.previous_waypoint = self.current_waypoint
        self.current_waypoint = self.choices[choice_index]

    def draw_choices(self):
        for i, choice in enumerate(self.choices):
            location = choice.transform.location
            self.debug.draw_point(location,
                                  size=0.3,
                                  life_time=self._waypoint_life_time,
                                  color=self._blue)
            self.debug.draw_string(location,
                                   text=str(i),
                                   color=self._white,
                                   life_time=self._waypoint_life_time)

    def draw_current_waypoint(self):
        if self.previous_waypoint is None:
            self.previous_waypoint = self.current_waypoint

        if self.current_waypoint == self.previous_waypoint:
            self._waypoint_equal_count += 1
        else:
            self._waypoint_equal_count = 1

        additional_z_axis = 0.1 * self._waypoint_equal_count

        current_location = self.current_waypoint.transform.location
        new_location = carla.Location(x=current_location.x,
                                      y=current_location.y,
                                      z=current_location.z + additional_z_axis)

        self.debug.draw_point(new_location,
                              size=0.5,
                              life_time=self._waypoint_life_time,
                              color=self._green)


def connect_to_carla(load_new_world=True):
    host = 'localhost'
    port = 2000
    map = 'ait_v4'

    client = carla.Client(host, port)
    client.set_timeout(20.0)

    if load_new_world:
        return client.load_world(map)
    else:
        return client.get_world()


def get_starting_waypoint(world):
    map = world.get_map()

    # TOOD: do whatever it takes to get your starting waypoint here
    # 9 is by 7-11, 11 is at the intersection
    vehicle_spawn_points = list(map.get_spawn_points())
    reference_starting_waypoint = map.get_waypoint(vehicle_spawn_points[11].location)

    next_points = reference_starting_waypoint.next(2)
    starting_waypoint = next_points[0].get_left_lane()

    return starting_waypoint


class Controller:
    def __init__(self, travel_manager):
        self.travel_manager = travel_manager
        self.term = Terminal()

        self.setup()

    def setup(self):
        self.travel_manager.draw_current_waypoint()

    def run(self):
        with self.term.cbreak():
            while True:
                travel_propmt = 'Press UP to call next, DOWN to call previous:'
                travel_key = self._receive_keycode(travel_propmt, [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_EXIT])

                if travel_key.code == curses.KEY_UP:
                    self.travel_manager.call_next()
                elif travel_key.code == curses.KEY_DOWN:
                    self.travel_manager.call_previous()
                elif travel_key.code == curses.KEY_EXIT:
                    break
                else:
                    raise Exception(f'Unknown command {travel_key}')

                if self.travel_manager.have_choices():
                    # self.travel_manager.increase_distance_when_have_choices()
                    self.travel_manager.draw_choices()
                    
                    choices_len = len(self.travel_manager.choices)
                    choices_index = list(map(str, range(choices_len)))
                    select_choice_propmt = f'Type index of array (0-{len(choices_index)}) to select choice:'

                    choice_index = self._receive_text(select_choice_propmt,
                                                      list(map(str, range(choices_len))))
                elif self.travel_manager.have_one_choice():
                    choice_index = 0
                else:
                    print('does not have choice')
                    choice_index = None

                if choice_index is not None:
                    self.travel_manager.select_waypoint(int(choice_index))

                self.travel_manager.draw_current_waypoint()

    def _receive_keycode(self, prompt_text, possible_codes) -> str:
        assert len(possible_codes) > 0

        keycode = None
        while keycode is None or keycode.code not in possible_codes:
            print(prompt_text)
            keycode = self.term.inkey()

        return keycode

    def _receive_text(self, prompt_text, possible_values):
        assert len(possible_values) > 0

        text = None
        while text is None or text not in possible_values:
            char_key = self.term.inkey()
            text = str(char_key)

        return text


if __name__ == '__main__':
    carla_world = connect_to_carla(load_new_world=False)

    starting_waypoint = get_starting_waypoint(carla_world)
    travel_manager = TravelManager(carla_world, starting_waypoint, distance=6)
    
    controller = Controller(travel_manager)
    controller.run()
