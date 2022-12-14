import color_helpers
from typing import Optional
from enum import Enum
import pygame
import gym
from gym import spaces
from ctypes import c_int, POINTER
import ctypes
import numpy as np
import os
# Should be before pygame import (may be out of order because of autoformatting)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
c_helpers = ctypes.CDLL('./env_helpers.so')
# ai.get_action.restype = POINTER(c_int)


class MPSPEnv(gym.Env):
    """Environment for the Multi Port Shipping Problem"""

    def __init__(self, rows, columns, n_ports, render_mode: Optional[str] = None,):
        super(MPSPEnv, self).__init__()
        self.R = rows
        self.C = columns
        self.N = n_ports
        self.is_terminated = 0
        # Random number between 0 and 1.000.000
        self.c_seed = np.random.randint(1_000_000)
        self.exponential_constant = 0.5
        self.exponential_multiplier = 10

    def reset(self, transportation_matrix=None, seed=None):
        """Reset the state of the environment to an initial state"""
        if seed is not None:
            self.c_seed = seed
        self.loading_list = []
        self.transportation_matrix = self._get_mixed_transportation_matrix(
            self.N,
            self.c_seed
        )
        self.loading_list = self._get_loading_list(
            self.transportation_matrix, self.N
        )
        self.bay_matrix = (c_int * self.R * self.C)(
            0 for _ in range(self.R * self.C)
        )
        self.column_counts = (c_int * self.C)(0 for _ in range(self.C))
        # Initialize to max values
        self.min_value_per_column = (c_int * self.C)(
            self.R for _ in range(self.C)
        )
        self.port = 0
        self.reward = 0
        self.is_terminated = 0

    #     return self._get_observation()

    # def step(self, action):
    #     """Execute one time step within the environment

    #     Args:
    #         action (int): The action to be executed
    #         The first C actions are for adding containers
    #         The last C actions are for removing containers
    #     """
    #     assert not self.is_terminated, "Environment is terminated"

    #     should_add = action < self.C
    #     reward = 0

    #     if should_add:
    #         j = action
    #         i = self.R - self.column_counts[j] - 1

    #         assert self.column_counts[j] < self.R, "Cannot add containers to full columns"

    #         reward += self._add_container(i, j)
    #     else:
    #         j = action - self.C
    #         i = self.R - self.column_counts[j]

    #         assert self.column_counts[
    #             action - self.C
    #         ] > 0, "Cannot remove containers from empty columns"

    #         reward += self._remove_container(i, j)

    #     # Port is zero indexed
    #     self.is_terminated = self.port+1 == self.N

    #     info = {
    #         "mask": self.action_masks()
    #     }

    #     self.reward += reward

    #     return (
    #         self._get_observation(),
    #         reward,
    #         self.is_terminated,
    #         info
    #     )

    # def action_masks(self):
    #     """Returns a mask for the actions (True if the action is valid, False otherwise)."""

    #     # Masking out full columns
    #     add_mask = (
    #         self.column_counts < self.R
    #         if self.virtual_R is None
    #         else self.column_counts < self.virtual_R
    #     )

    #     if self.virtual_C is not None:
    #         # Masking out columns that are not accessible
    #         add_mask = np.logical_and(
    #             add_mask,
    #             # Can only use first virtual_C columns
    #             np.arange(self.C) < self.virtual_C
    #         )

    #     # Masking out empty columns
    #     remove_mask = self.column_counts > 0

    #     mask = np.concatenate((add_mask, remove_mask), dtype=np.int8)

    #     return mask

    # def close(self):
    #     pass

    # def print(self):
    #     print(f'Port: {self.port}')
    #     print('Bay matrix:')
    #     print(self.bay_matrix)
    #     print('Transportation matrix:')
    #     print(self.transportation_matrix)
    #     print('Column counts:')
    #     print(self.column_counts)
    #     print('Min value per column:')
    #     print(self.min_value_per_column)
    #     print('Loading list:')
    #     print(self.loading_list)

    # def render(self, mode='human'):

    #     return self._render_human(mode)

    # def _render_human(self, mode='human'):

    #     # Initialise screen
    #     H, W = 400, 600
    #     pygame.font.init()
    #     if self.screen is None and mode == "human":
    #         # Make a dict of gradient colors
    #         pygame.init()
    #         pygame.display.init()
    #         self.screen = pygame.display.set_mode((W, H))

    #     if self.colors is None:
    #         # self.colors = {i: color for i, color in enumerate(helpers.get_color_gradient('#A83279', '#D38312', self.N))}
    #         # Random distinct light colors without helper
    #         self.colors = {
    #             i: tuple(np.random.randint(128, 255, size=3))
    #             for i in range(self.N)
    #         }

    #     # Fill background
    #     self.surface = pygame.Surface((W, H))
    #     # self.surface = self.surface.convert()
    #     self.surface.fill((255, 255, 255))

    #     PADDING = 20

    #     # Render current port
    #     self._render_text(f'Port: {self.port}, Reward: {self.reward}', pos=(
    #         W/2, PADDING), font_size=text_type.HEADLINE)

    #     # Render the bay matrix and transportation matrix
    #     CELL_SIZE = 20
    #     bay_frame_size = (self.C * CELL_SIZE, self.R * CELL_SIZE)
    #     transport_frame_size = (self.N * CELL_SIZE, self.N * CELL_SIZE)

    #     width_sum = bay_frame_size[0] + transport_frame_size[0] + PADDING
    #     bay_x = W/2-width_sum/2
    #     transport_x = bay_x + bay_frame_size[0] + PADDING

    #     self._render_bay(cell_size=CELL_SIZE, pos=(
    #         bay_x, PADDING*3))
    #     self._render_transportation_matrix(
    #         cell_size=CELL_SIZE, pos=(transport_x, PADDING*3))

    #     # Render the container explanation
    #     self._render_container_explanation(cell_size=CELL_SIZE, pos=(
    #         transport_x, PADDING*5 + transport_frame_size[1]))
    #     self._render_action_probabilities(cell_size=CELL_SIZE, pos=(
    #         bay_x, PADDING*5 + bay_frame_size[1]))

    #     if mode == "human":
    #         # Blit everything to the screen
    #         pygame.event.pump()
    #         self.screen.fill(0)
    #         self.screen.blit(self.surface, (0, 0))
    #         pygame.display.flip()
    #     elif mode == "rgb_array":
    #         return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))
    #     else:
    #         raise NotImplementedError

    # def _render_action_probabilities(self, cell_size, pos=(0, 0)):
    #     """Renders the action probabilities"""
    #     x, y = pos

    #     if self.probs is None or self.prev_action is None:
    #         return

    #     # Draw to rows of probabilities. One for adding and one for removing.
    #     # Draw a box for each action. Remove are below the add boxes.
    #     # Color the box darker green depending on the probability
    #     # If it is the action, make the border thicker
    #     gradient = color_helpers.get_color_gradient('#dd3e54', '#6be585', 100)

    #     probs = self.probs.detach().cpu().numpy().squeeze()

    #     for i, prob in enumerate(probs):
    #         prob = prob*100
    #         if not self.action_mask[i]:
    #             color = 'white'
    #         else:
    #             color = gradient[int(prob)]

    #         # Draw the colored box
    #         pygame.draw.rect(
    #             self.surface,
    #             color,
    #             (
    #                 x + i * cell_size if i < self.C else x +
    #                 (i - self.C) * cell_size,
    #                 y if i < self.C else y + cell_size,
    #                 cell_size,
    #                 cell_size
    #             ),
    #         )

    #         # Draw the border if it is the action
    #         if i == self.prev_action:
    #             pygame.draw.rect(
    #                 self.surface,
    #                 (0, 0, 0),
    #                 (
    #                     x + i * cell_size if i < self.C else x +
    #                     (i - self.C) * cell_size,
    #                     y if i < self.C else y + cell_size,
    #                     cell_size,
    #                     cell_size
    #                 ),
    #                 2
    #             )

    #         self._render_text(
    #             f'{np.round(prob,1)}',
    #             pos=(
    #                 x + i * cell_size + cell_size/2 if i < self.C else x +
    #                 (i - self.C) * cell_size + cell_size/2,
    #                 y + cell_size/2 if i < self.C else y + cell_size + cell_size/2
    #             ),
    #             font_size=text_type.CELL
    #         )

    # def _render_container_explanation(self, cell_size, pos=(0, 0)):
    #     """Renders the container explanation"""
    #     x, y = pos

    #     for i, color in enumerate(self.colors.values()):

    #         self._render_text(f'{i}', pos=(x + cell_size*i + cell_size/2, y))

    #         text_offset = 15
    #         pygame.draw.rect(
    #             self.surface,
    #             color,
    #             (
    #                 x + i * cell_size,
    #                 y + text_offset,
    #                 cell_size,
    #                 cell_size
    #             ),
    #         )

    # def _render_bay(self, cell_size, pos=(0, 0)):
    #     """Renders the bay matrix"""

    #     x, y = pos

    #     center_x, center_y = (x + self.C * cell_size / 2,
    #                           y + self.R * cell_size / 2)
    #     self._render_text(f'Bay', pos=(center_x, y))

    #     text_offset = 15
    #     blocking_containers = self._get_blocking()

    #     # Draw the grid lines and the containers
    #     for i in range(self.R):
    #         for j in range(self.C):

    #             # Draw the grid lines
    #             pygame.draw.rect(
    #                 self.surface,
    #                 (0, 0, 0),
    #                 (
    #                     x + j * cell_size,
    #                     y + text_offset + i * cell_size,
    #                     cell_size,
    #                     cell_size
    #                 ),
    #                 1
    #             )

    #             # Draw the containers
    #             container = self.bay_matrix[i, j]
    #             if container > 0:
    #                 pygame.draw.rect(
    #                     self.surface,
    #                     self.colors[container],
    #                     (
    #                         x + j * cell_size,
    #                         y + text_offset + i * cell_size,
    #                         cell_size,
    #                         cell_size
    #                     )
    #                 )

    #             # Draw a little B for all blocking containers
    #             if blocking_containers[i, j] == 1:
    #                 self._render_text(
    #                     'B',
    #                     pos=(x + j * cell_size + cell_size*3/4, y +
    #                          text_offset + i * cell_size + cell_size/4),
    #                     font_size=13
    #                 )

    # def _render_transportation_matrix(self, cell_size, pos=(0, 0)):
    #     """Renders the transportation matrix"""

    #     x, y = pos

    #     center_x, center_y = (x + self.N * cell_size / 2,
    #                           y + self.N * cell_size / 2)
    #     self._render_text(f'Transportation', pos=(center_x, y))

    #     text_offset = 15
    #     # Draw the grid lines and the containers
    #     for i in range(self.N):
    #         for j in range(self.N):
    #             # Draw the grid lines
    #             pygame.draw.rect(
    #                 self.surface,
    #                 (0, 0, 0),
    #                 (
    #                     x + j * cell_size,
    #                     y + text_offset + i * cell_size,
    #                     cell_size,
    #                     cell_size
    #                 ),
    #                 1
    #             )

    #             # Draw the containers
    #             count = self.transportation_matrix[i, j]
    #             if count > 0:
    #                 pygame.draw.rect(
    #                     self.surface,
    #                     self.colors[j],
    #                     (
    #                         x + j * cell_size,
    #                         y + text_offset + i * cell_size,
    #                         cell_size,
    #                         cell_size
    #                     )
    #                 )

    #                 self._render_text(
    #                     f'{count}',
    #                     pos=(
    #                         x + j * cell_size + cell_size/2, y + text_offset + i * cell_size + cell_size/2
    #                     ),
    #                     font_size=text_type.CELL
    #                 )

    # def _render_text(self, text, pos=(0, 0), font_size: text_type = text_type.SUBHEADLINE):
    #     """Renders the text"""

    #     if isinstance(font_size, text_type):
    #         font_size = font_size.value

    #     font = pygame.font.Font(pygame.font.get_default_font(), font_size)
    #     text_surface = font.render(text, True, (10, 10, 10))
    #     text_rect = text_surface.get_rect(center=pos)
    #     self.surface.blit(text_surface, text_rect)

    # def _remove_container(self, i, j):
    #     """Removes container from bay and returns delta reward"""

    #     # Update state
    #     container = self.bay_matrix[i, j]
    #     self.bay_matrix[i, j] = 0

    #     # Add to beginning of loading list
    #     if self.loading_list[0][1] == container:
    #         self.loading_list[0][0] += 1
    #     else:
    #         self.loading_list.insert(0, [1, container])

    #     # Check if min_value_per_column needs to be checked/updated
    #     if container == self.min_value_per_column[j]:
    #         self.min_value_per_column[j] = self.get_min_in_column(j)

    #     self.transportation_matrix[self.port, container] += 1
    #     self.column_counts[j] -= 1

    #     # Penalize shifting containers
    #     return -1

    # def _add_container(self, i, j):
    #     """Adds container to bay and returns delta reward"""

    #     delta_reward = 0
    #     self.column_counts[j] += 1

    #     # Find last destination container
    #     container = self.loading_list[0][1]

    #     # Update min_value_per_column
    #     self.min_value_per_column[j] = min(
    #         self.min_value_per_column[j],
    #         container
    #     )

    #     # Update state
    #     self.bay_matrix[i, j] = container
    #     self.transportation_matrix[self.port, container] -= 1

    #     # Either:
    #     # Remove the first container in the loading list
    #     # Decrease the count of the first container in the loading list
    #     if self.loading_list[0][0] == 1:
    #         self.loading_list.pop(0)
    #     else:
    #         self.loading_list[0][0] -= 1

    #     # Check if container is blocking (there exists a container in the same column with a higher destination)
    #     # If so, penalize
    #     if self.min_value_per_column[j] < container:
    #         delta_reward -= 1

    #     # Sail along for every zero-row
    #     while np.sum(self.transportation_matrix[self.port]) == 0:
    #         self.port += 1
    #         self._offload_containers()
    #         if self.port + 1 == self.N:
    #             break

    #     return delta_reward

    # def _create_image_array(self, screen, size):
    #     scaled_screen = pygame.transform.smoothscale(screen, size)
    #     return np.transpose(
    #         np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
    #     )

    # def _get_blocking(self):
    #     """Returns a matrix of blocking containers (1 if blocking, 0 otherwise)"""
    #     blocking_containers = np.zeros((self.R, self.C), dtype=np.int32)

    #     for j in range(self.C):
    #         min_in_column = np.inf
    #         for i in range(self.R-1, -1, -1):
    #             if self.bay_matrix[i, j] == 0:
    #                 break
    #             if self.bay_matrix[i, j] < min_in_column:
    #                 min_in_column = self.bay_matrix[i, j]
    #             if self.bay_matrix[i, j] > min_in_column:
    #                 blocking_containers[i, j] = 1

    #     return blocking_containers

    # def _offload_containers(self):
    #     """Offloads containers to the port, updates the transportation matrix and returns the number of shifts"""
    #     n_blocking_containers = 0

    #     for j in range(self.C):
    #         offloading_column = False
    #         for i in range(self.R-1, -1, -1):
    #             # We reached top of stack
    #             if self.bay_matrix[i, j] == 0:
    #                 break

    #             # If true, we must offload this container and all containers above it
    #             if self.bay_matrix[i, j] == self.port:
    #                 offloading_column = True

    #             if not offloading_column:
    #                 continue

    #             if self.bay_matrix[i, j] != self.port:
    #                 n_blocking_containers += 1
    #                 # Add container back into transportation matrix
    #                 destination_port = self.bay_matrix[i, j]
    #                 self.transportation_matrix[
    #                     self.port,
    #                     destination_port
    #                 ] += 1

    #             self.bay_matrix[i, j] = 0
    #             self.column_counts[j] -= 1

    #     # Rebuild min_value_per_column, as we dont know which containers were removed
    #     self.min_value_per_column = np.array([
    #         self.get_min_in_column(i) for i in range(self.C)
    #     ])

    #     # Rebuild load list
    #     self.loading_list = []
    #     self._get_loading_list()

    #     return n_blocking_containers

    # def get_min_in_column(self, j):
    #     """Returns the minimum value in column j (excluding zeros). If all values are zero, returns max int"""
    #     non_zero_values = self.bay_matrix[:, j][self.bay_matrix[:, j] > 0]
    #     if len(non_zero_values) == 0:
    #         return np.iinfo(np.int32).max
    #     else:
    #         return np.min(non_zero_values)

    # def _get_observation(self):
    #     if len(self.loading_list) == 0:
    #         # Last state, so no block
    #         will_block = np.zeros(self.C, dtype=np.int32)
    #         padded_loading_list = np.zeros(
    #             self.observation_space['loading_list'].shape,
    #         )
    #     else:
    #         next_container = self.loading_list[0][1]
    #         will_block = self.min_value_per_column < next_container
    #         padded_loading_list = np.pad(
    #             self.loading_list,
    #             # Pad with zeros to correct shape
    #             (
    #                 (0, self.observation_space['loading_list'].shape[0] - \
    #                  len(self.loading_list)),
    #                 (0, 0)
    #             ),
    #             'constant'
    #         )

    #     return {
    #         'bay_matrix': self.bay_matrix,
    #         'port': [self.port],
    #         'will_block': will_block,
    #         'loading_list': padded_loading_list,
    #         'loading_list_length': [len(self.loading_list)],
    #     }

    # def _get_mixed_transportation_matrix(self, N):
    #     """Generates a feasible transportation matrix (mixed distance)"""
    #     ordering = []

    #     for i in range(N-1):
    #         ordering.append(np.arange(N-1, i, -1))

    #     return self._get_transportation_matrix(N, ordering)

    # def _get_short_distance_transportation_matrix(self, N):
    #     """Generates a feasible transportation matrix (short distance)"""
    #     ordering = []

    #     for i in range(N-1):
    #         ordering.append(np.arange(i+1, N))

    #     return self._get_transportation_matrix(N, ordering)

    # def _get_long_distance_transportation_matrix(self, N):
    #     """Generates a feasible transportation matrix (long distance)"""
    #     ordering = []

    #     for i in range(N-1):
    #         ordering.append(np.arange(N-1, i, -1))

    #     return self._get_transportation_matrix(N, ordering)

    # def _get_loading_list(self):
    #     """Create loading list from transportation matrix"""
    #     for i in range(self.N-1):
    #         for j in range(self.N-1, i, -1):
    #             count = self.transportation_matrix[i, j]

    #             if count == 0:
    #                 continue

    #             container = j
    #             self.loading_list.append([count, container])

    # def _get_transportation_matrix(self, N, ordering):
    #     """Generates a feasible transportation matrix (short distance)

    #     Args:
    #         N (int): Number of ports
    #         ordering (list): List of lists of what ports to add destination containers to first
    #     """
    #     output = np.zeros((N, N), dtype=np.int32)
    #     bay_capacity = (
    #         self.capacity
    #         if self.virtual_Capacity is None else
    #         self.virtual_Capacity
    #     )

    #     for i in range(N-1):
    #         for j in ordering[i]:
    #             output[i, j] = min(
    #                 np.random.exponential(
    #                     exponential_constant
    #                 ) * exponential_multiplier,
    #                 bay_capacity
    #             )
    #             bay_capacity -= output[i, j]

    #         if bay_capacity > 0:
    #             # Make sure that ship is fully loaded
    #             output[i, np.random.choice(ordering[i])] += bay_capacity
    #             bay_capacity = 0

    #         # Offloaded at port
    #         for h in range(i+1):
    #             bay_capacity += output[h, i+1]

    #     return output
