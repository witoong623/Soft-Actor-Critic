import abc
import functools
import operator
import random


class MapCheckpointManager(abc.ABC):
    def __init__(self, route_waypoint_len, repeat_section_threshold=5) -> None:
        self._route_waypoint_len = route_waypoint_len
        self._repeat_count_threshold = repeat_section_threshold
        self._repeat_count = 0

        self._checkpoint_index = 0
        self._next_checkpoint_index = 0
        self._start_index = 0

    @abc.abstractclassmethods
    def update_checkpoint(self, current_waypoint_index):
        raise NotImplementedError()

    @abc.abstractclassmethods
    def get_spawn_point_index(self):
        raise NotImplementedError()


START = 0
END = 1
FREQUENCY = 2


class Town7CheckpointManager(MapCheckpointManager):
    def __init__(self, route_waypoint_len=592, repeat_section_threshold=5,
                 initial_checkpoint=0, **kwargs) -> None:
        super().__init__(route_waypoint_len, repeat_section_threshold)

        self._checkpoint_index = initial_checkpoint
        self._start_index = initial_checkpoint

        self._in_random_spawn_point = False
        # TODO: implement this
        self._reached_last_waypoint_index = False
        self.round_spawn_idx = 0

        # (start, end, checkpoint frequency)
        self.sections_indexes = [(0, 140, 35), (143, 173, 30), (176, route_waypoint_len - 1, 35)]
        self.sections_start = [s[START] for s in self.sections_indexes]
        self.sections_end = [s[END] for s in self.sections_indexes]
        self.sections_frequency = [s[FREQUENCY] for s in self.sections_indexes]
        self.sections_ends = [140, 141, 142, 173, 174, 175, 591]

        def get_all_indexes(start, end, step):
            indexes = []
            idx = start

            while not indexes or indexes[-1] < end:
                idx = max(idx + step, end)
                indexes.append(idx)

            return indexes

        self.all_spawn_indexes = functools.reduce(operator.concat,
                                                  [get_all_indexes(*sec) for sec in self.sections_indexes])

        if initial_checkpoint < self.sections_end[0]:
            frequency = self.sections_indexes[0][FREQUENCY]
        elif initial_checkpoint < self.sections_end[1]:
            frequency = self.sections_indexes[1][FREQUENCY]
        elif initial_checkpoint < self.sections_end[2]:
            frequency = self.sections_indexes[2][FREQUENCY]

        self._next_checkpoint_index = self._checkpoint_index + frequency
        if self._next_checkpoint_index > route_waypoint_len - 1:
            self._next_checkpoint_index = 0

    def update_checkpoint(self, current_waypoint_index):
        start_index, end_index, frequency = self._get_current_section_info()

        if current_waypoint_index == end_index:
            latest_milestone_index = end_index
        else:
            # get the most recent progress in term of index complete
            # discard any progress that doesn't reach checkpoint index
            latest_milestone_index = (((current_waypoint_index - start_index) // frequency) * frequency) + start_index

        if latest_milestone_index >= self._next_checkpoint_index:
            self._repeat_count += 1

            if self._repeat_count >= self._repeat_count_threshold:
                if self._is_done_repeating(start_index):
                    if self._should_start_in_next_section(end_index):
                        self._update_start_next_section()
                    else:
                        self._update_in_same_section()
                else:
                    self._checkpoint_index = start_index

                self._repeat_count = 0

    def _get_next_section_start_and_frequency(self, end_of_section):
        end_idx = self.sections_end.index(end_of_section)
        next_start = self.sections_start[(end_idx + 1) % len(self.sections_start)]
        next_frequency = self.sections_frequency[(end_idx + 1) % len(self.sections_frequency)]
        return next_start, next_frequency

    def _get_current_section_info(self):
        s1, s2, s3 = self.sections_indexes

        selected_index = -1
        if s1[START] <= self._start_index <= s1[END]:
            selected_index = 0
        elif s2[START] <= self._start_index <= s2[END]:
            selected_index = 1
        else:
            # s3
            selected_index = 2

        return self.sections_indexes[selected_index]

    def _update_start_next_section(self, end_index, frequency):
        self._checkpoint_index, frequency = self._get_next_section_start_and_frequency(end_index)
        self._next_checkpoint_index = self._checkpoint_index + frequency

    def _update_in_same_section(self, end_index, frequency):
        self._checkpoint_index = self._next_checkpoint_index
        self._next_checkpoint_index = min(self._checkpoint_index + frequency, end_index)

    def _is_done_repeating(self, start_index):
        return self._checkpoint_index == start_index

    def _should_start_in_next_section(self, end_index):
        return self._next_checkpoint_index >= end_index

    def _get_random_spawn_point(self):
        start_original = random.random() >= 0.4
        if start_original:
            self._in_random_spawn_point = False
            spawn_idx = self._checkpoint_index
        else:
            self._in_random_spawn_point = True

            if random.random() >= 0.3 or self._checkpoint_index in self.sections_start:
                # random start in the same section
                spawn_idx = self._checkpoint_index + (random.randint(5, 20) // 2 * 2)
            else:
                # random start at any point before current checkpoint
                lower_bound = 0
                for start, end in zip(self.sections_start, self.sections_ends):
                    if start <= self._checkpoint_index < end:
                        lower_bound = start
                        break

                spawn_idx = random.randint(lower_bound, self._checkpoint_index)

        return spawn_idx
