import os
import gym
import ctypes
import pygame
from ctypes import POINTER, c_int, c_double, Structure
from typing import Optional
from enum import Enum
from gym import spaces
import color_helpers
import numpy as np
# Should be before pygame import (may be out of order because of autoformatting)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class State(Structure):
    _fields_ = [
        ("N", c_int),
        ("R", c_int),
        ("C", c_int),
        ("port", c_int),
        ("bay_matrix", POINTER(c_int)),
        ("transportation_matrix", POINTER(c_int)),
        ("loading_list", POINTER(c_int)),
        ("column_counts", POINTER(c_int)),
        ("min_container_per_column", POINTER(c_int)),
        ("containers_per_port", POINTER(c_int)),
        ("mask", POINTER(c_int)),
        ("loading_list_length", c_int),
        ("loading_list_padded_length", c_int),
        ("is_terminal", c_int),
        ("last_reward", c_int),
        ("last_action", c_int),
        ("sum_reward", c_int),
        ("remove_restrictions", c_int),
    ]


c_helpers = ctypes.CDLL('./env_helpers.so')
c_helpers.get_random_state.restype = POINTER(State)
c_helpers.get_state_from_transportation_matrix.restype = POINTER(State)
c_helpers.get_blocking.restype = POINTER(c_int)
c_helpers.free_blocking.restype = None
c_helpers.step.restype = None
c_helpers.free_state.restype = None


class text_type(Enum):
    CELL = 8
    HEADLINE = 36
    SUBHEADLINE = 28


remove_lookup = {
    'remove_all': 0,
    'remove_only_when_blocking': 1,
    'no_remove': 2
}


class MPSPEnv(gym.Env):
    """Environment for the Multi Port Shipping Problem"""

    def __init__(self, rows, columns, n_ports, remove_restrictions, render_mode: Optional[str] = None):
        super(MPSPEnv, self).__init__()
        self.R = rows
        self.C = columns
        self.N = n_ports
        self.exponential_constant = 0.25  # Also called 'lambda'
        self.screen = None
        self.colors = None
        self.probs = None
        self.remove_restrictions = remove_lookup[remove_restrictions]
        self.render_mode = render_mode
        self.metadata = {
            "render.modes": [
                "human",
                "rgb_array",
            ],
        }

        self.action_space = spaces.Discrete(2 * self.C)
        bay_matrix_def = spaces.Box(
            low=0,
            high=self.N-1,
            shape=(self.R, self.C),
            dtype=np.int32
        )
        port_def = spaces.Box(
            low=0,
            high=self.N-1,
            shape=(1,),
            dtype=np.int32
        )
        loading_list_def = spaces.Box(
            low=0,
            # The count can be at most be C*R
            high=self.C*self.R,
            # Has length n(n-1)/2 because we only need to store the upper triangle
            # Has width 2 because we need to store the count and destination
            shape=(self.N*(self.N-1) // 2, 2),
            dtype=np.int32
        )
        loading_list_length_def = spaces.Box(
            low=0,
            high=self.N*(self.N-1) / 2,
            shape=(1,),
            dtype=np.int32
        )
        self.observation_space = spaces.Dict({
            'bay_matrix': bay_matrix_def,
            'port': port_def,
            'loading_list': loading_list_def,
            'loading_list_length': loading_list_length_def,
        })

    def seed(self, seed=None):
        np.random.seed(seed)
        if seed is None:
            self.c_seed = np.random.randint(
                0,
                np.iinfo(np.int32).max,
            )
        else:
            self.c_seed = seed

    def reset(self, transportation_matrix=None, seed=None):
        """Reset the state of the environment to an initial state"""
        self.seed(seed)

        # Caller should free the state once it is terminated
        if transportation_matrix is None:
            self.state = c_helpers.get_random_state(
                c_int(self.N),
                c_int(self.R),
                c_int(self.C),
                c_double(self.exponential_constant),
                c_int(self.c_seed),
                c_int(self.remove_restrictions)
            )
        else:
            assert transportation_matrix.dtype == np.int32, "Transportation matrix must be of type np.int32"
            self.state = c_helpers.get_state_from_transportation_matrix(
                c_int(self.N),
                c_int(self.R),
                c_int(self.C),
                transportation_matrix.ctypes.data_as(POINTER(c_int)),
                c_int(self.remove_restrictions),
            )

        # ----- NOTE: The following numpy arrays are views of the underlying C arrays (not a copy)
        self.bay_matrix = np.ctypeslib.as_array(
            self.state.contents.bay_matrix,
            shape=(self.R, self.C),
        )
        self.loading_list = np.ctypeslib.as_array(
            self.state.contents.loading_list,
            shape=(self.state.contents.loading_list_padded_length, 2),
        )
        self.action_mask = np.ctypeslib.as_array(
            self.state.contents.mask,
            shape=(2 * self.C,),
        )
        self.transportation_matrix = np.ctypeslib.as_array(
            self.state.contents.transportation_matrix,
            shape=(self.N, self.N),
        )
        self.column_counts = np.ctypeslib.as_array(
            self.state.contents.column_counts,
            shape=(self.C,),
        )
        self.min_container_per_column = np.ctypeslib.as_array(
            self.state.contents.min_container_per_column,
            shape=(self.C,),
        )
        self.containers_per_port = np.ctypeslib.as_array(
            self.state.contents.containers_per_port,
            shape=(self.N,),
        )
        # -----

        return self._get_observation()

    def step(self, action):
        """Execute one time step within the environment

        Args:
            action (int): The action to be executed
            The first C actions are for adding containers
            The last C actions are for removing containers
        """
        c_helpers.step(c_int(action), self.state)
        is_terminal = self.state.contents.is_terminal
        reward = self.state.contents.last_reward
        observation = self._get_observation()

        return (
            observation,
            reward,
            is_terminal,
            {}
        )

    def action_masks(self):
        """Returns a mask for the actions (True if the action is valid, False otherwise)."""
        return self.action_mask.copy()

    def close(self):
        """Free the memory allocated in C"""
        print('Closed environment')
        c_helpers.free_state(self.state)

    def print(self):
        print('Min container per column:')
        print(self.min_container_per_column)
        print('Loading list:')
        print(self.loading_list)
        print('Loading list length:')
        print(self.state.contents.loading_list_length)
        print("Reward:")
        print(self.state.contents.last_reward)
        print("Sum reward:")
        print(self.state.contents.sum_reward)
        print("Is terminated:")
        print(self.state.contents.is_terminal)
        print("Containers per port:")
        print(self.containers_per_port)
        print(f'Port: {self.state.contents.port}')
        print("Mask:")
        print(self.action_mask)
        print('Transportation matrix:')
        print(self.transportation_matrix)
        print('Bay matrix:')
        print(self.bay_matrix)
        print('Column counts:')
        print(self.column_counts)
        print()

    def render(self, mode='human'):

        return self._render_human(mode)

    def _render_human(self, mode='human'):

        # Initialise screen
        H, W = 400, 600
        pygame.font.init()
        if self.screen is None and mode == "human":
            # Make a dict of gradient colors
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((W, H))

        if self.colors is None:
            # self.colors = {i: color for i, color in enumerate(helpers.get_color_gradient('#A83279', '#D38312', self.N))}
            # Random distinct light colors without helper
            self.colors = {
                i: tuple(np.random.randint(128, 255, size=3))
                for i in range(self.N)
            }

        # Fill background
        self.surface = pygame.Surface((W, H))
        # self.surface = self.surface.convert()
        self.surface.fill((255, 255, 255))

        PADDING = 20

        # Render current port
        self._render_text(f'Port: {self.state.contents.port}, Reward: {self.state.contents.sum_reward}', pos=(
            W/2, PADDING), font_size=text_type.HEADLINE)

        # Render the bay matrix and transportation matrix
        CELL_SIZE = 20
        bay_frame_size = (self.C * CELL_SIZE, self.R * CELL_SIZE)
        transport_frame_size = (self.N * CELL_SIZE, self.N * CELL_SIZE)

        width_sum = bay_frame_size[0] + transport_frame_size[0] + PADDING
        bay_x = W/2-width_sum/2
        transport_x = bay_x + bay_frame_size[0] + PADDING

        self._render_bay(cell_size=CELL_SIZE, pos=(
            bay_x, PADDING*3))
        self._render_transportation_matrix(
            cell_size=CELL_SIZE, pos=(transport_x, PADDING*3))

        # Render the container explanation
        self._render_container_explanation(cell_size=CELL_SIZE, pos=(
            transport_x, PADDING*5 + transport_frame_size[1]))
        self._render_action_probabilities(cell_size=CELL_SIZE, pos=(
            bay_x, PADDING*5 + bay_frame_size[1]))

        if mode == "human":
            # Blit everything to the screen
            pygame.event.pump()
            self.screen.fill(0)
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))
        else:
            raise NotImplementedError

    def _render_action_probabilities(self, cell_size, pos=(0, 0)):
        """Renders the action probabilities"""
        x, y = pos

        if self.probs is None or self.state.contents.last_action == -1:
            return

        # Draw to rows of probabilities. One for adding and one for removing.
        # Draw a box for each action. Remove are below the add boxes.
        # Color the box darker green depending on the probability
        # If it is the action, make the border thicker
        gradient = color_helpers.get_color_gradient('#dd3e54', '#6be585', 100)

        probs = self.probs.detach().cpu().numpy().squeeze()

        for i, prob in enumerate(probs):
            prob = prob*100
            if not self.action_mask[i]:
                color = 'white'
            else:
                color = gradient[int(prob)]

            # Draw the colored box
            pygame.draw.rect(
                self.surface,
                color,
                (
                    x + i * cell_size if i < self.C else x +
                    (i - self.C) * cell_size,
                    y if i < self.C else y + cell_size,
                    cell_size,
                    cell_size
                ),
            )

            # Draw the border if it is the action
            if i == self.state.contents.last_action:
                pygame.draw.rect(
                    self.surface,
                    (0, 0, 0),
                    (
                        x + i * cell_size if i < self.C else x +
                        (i - self.C) * cell_size,
                        y if i < self.C else y + cell_size,
                        cell_size,
                        cell_size
                    ),
                    2
                )

            self._render_text(
                f'{np.round(prob,1)}',
                pos=(
                    x + i * cell_size + cell_size/2 if i < self.C else x +
                    (i - self.C) * cell_size + cell_size/2,
                    y + cell_size/2 if i < self.C else y + cell_size + cell_size/2
                ),
                font_size=text_type.CELL
            )

    def _render_container_explanation(self, cell_size, pos=(0, 0)):
        """Renders the container explanation"""
        x, y = pos

        for i, color in enumerate(self.colors.values()):

            self._render_text(f'{i}', pos=(x + cell_size*i + cell_size/2, y))

            text_offset = 15
            pygame.draw.rect(
                self.surface,
                color,
                (
                    x + i * cell_size,
                    y + text_offset,
                    cell_size,
                    cell_size
                ),
            )

    def _render_bay(self, cell_size, pos=(0, 0)):
        """Renders the bay matrix"""

        x, y = pos

        center_x, center_y = (x + self.C * cell_size / 2,
                              y + self.R * cell_size / 2)
        self._render_text(f'Bay', pos=(center_x, y))

        text_offset = 15
        # blocking_containers = self._get_blocking()
        blocking_containers = np.zeros((self.R, self.C))

        # Draw the grid lines and the containers
        for i in range(self.R):
            for j in range(self.C):

                # Draw the grid lines
                pygame.draw.rect(
                    self.surface,
                    (0, 0, 0),
                    (
                        x + j * cell_size,
                        y + text_offset + i * cell_size,
                        cell_size,
                        cell_size
                    ),
                    1
                )

                # Draw the containers
                container = self.bay_matrix[i, j]
                if container > 0:
                    pygame.draw.rect(
                        self.surface,
                        self.colors[container],
                        (
                            x + j * cell_size,
                            y + text_offset + i * cell_size,
                            cell_size,
                            cell_size
                        )
                    )

                # Draw a little B for all blocking containers
                if blocking_containers[i, j] == 1:
                    self._render_text(
                        'B',
                        pos=(x + j * cell_size + cell_size*3/4, y +
                             text_offset + i * cell_size + cell_size/4),
                        font_size=13
                    )

        # Free the memory
        c_helpers.free_blocking(
            blocking_containers.ctypes.data_as(POINTER(c_int))
        )

    def _render_transportation_matrix(self, cell_size, pos=(0, 0)):
        """Renders the transportation matrix"""

        x, y = pos

        center_x, center_y = (x + self.N * cell_size / 2,
                              y + self.N * cell_size / 2)
        self._render_text(f'Transportation', pos=(center_x, y))

        text_offset = 15
        # Draw the grid lines and the containers
        for i in range(self.N):
            for j in range(self.N):
                # Draw the grid lines
                pygame.draw.rect(
                    self.surface,
                    (0, 0, 0),
                    (
                        x + j * cell_size,
                        y + text_offset + i * cell_size,
                        cell_size,
                        cell_size
                    ),
                    1
                )

                # Draw the containers
                count = self.transportation_matrix[i, j]
                if count > 0:
                    pygame.draw.rect(
                        self.surface,
                        self.colors[j],
                        (
                            x + j * cell_size,
                            y + text_offset + i * cell_size,
                            cell_size,
                            cell_size
                        )
                    )

                    self._render_text(
                        f'{count}',
                        pos=(
                            x + j * cell_size + cell_size/2, y + text_offset + i * cell_size + cell_size/2
                        ),
                        font_size=text_type.CELL
                    )

    def _render_text(self, text, pos=(0, 0), font_size: text_type = text_type.SUBHEADLINE):
        """Renders the text"""

        if isinstance(font_size, text_type):
            font_size = font_size.value

        font = pygame.font.Font(pygame.font.get_default_font(), font_size)
        text_surface = font.render(text, True, (10, 10, 10))
        text_rect = text_surface.get_rect(center=pos)
        self.surface.blit(text_surface, text_rect)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def _get_blocking(self):
        """
        Returns a matrix of blocking containers (1 if blocking, 0 otherwise)
        Caller should free the memory
        """
        return np.ctypeslib.as_array(
            c_helpers.get_blocking(self.state),
            shape=(self.R, self.C)
        )

    def _get_observation(self):
        return {
            # Copy since the ndarrays are views
            'bay_matrix': self.bay_matrix.copy(),
            'port': [self.state.contents.port],
            'loading_list': self.loading_list.copy(),
            'loading_list_length': [self.state.contents.loading_list_length],
        }
