#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "env_helpers.h"
#include <assert.h>

// Returns a random number from an exponential distribution with parameter lambda
double ran_expo(double lambda)
{
    double u = rand() / (RAND_MAX + 1.0);
    return -log(1 - u) / lambda;
}

// Returns a random integer in the range [min, max)
int random_int(int min, int max)
{
    return min + rand() / (RAND_MAX / (max - min) + 1);
}

// Shuffles an array of integers
void shuffle(int *array, int n)
{
    if (n > 1)
    {
        for (int i = 0; i < n - 1; i++)
        {
            int j = random_int(i, n);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// Returns an array of integers from start to end - 1 in random order
int *get_ordering(int start, int end)
{
    int *ordering = malloc((end - start) * sizeof(int));
    for (int i = 0; i < end - start; i++)
    {
        ordering[i] = i + start;
    }
    shuffle(ordering, end - start);
    return ordering;
}

// Returns an array of zeros of length n
int *get_zeros(int n)
{
    int *zeros = calloc(n, sizeof(int));
    return zeros;
}
// Returns a binary array of length 2*c
// The first c elements are the add mask
// The last c elements are the remove mask
void insert_mask(struct state *state)
{
    for (int j = 0; j < state->C; j++)
    {
        // Add mask
        state->mask[j] = state->column_counts[j] < state->R;
        // Remove mask
        state->mask[j + state->C] = state->column_counts[j] > 0;
    }
}

// Generates the loading list and inserts it into the state
void insert_loading_list(struct state *state)
{
    int list_index = 0;

    // Iterate over the upper triangle of the transportation matrix
    for (int i = 0; i < state->N - 1; i++)
    {
        for (int j = state->N - 1; j >= i + 1; j--)
        {
            int matrix_index = i * state->N + j;
            int count = state->transportation_matrix[matrix_index];
            if (count > 0)
            {
                state->loading_list[2 * list_index] = count;
                state->loading_list[2 * list_index + 1] = j; // container
                list_index++;
            }
        }
    }

    state->loading_list = state->loading_list;
    state->loading_list_length = list_index;
}

// Left shifts the loading list (removes first element)
void left_shift_loading_list(struct state *state)
{
    for (int i = 0; i < state->loading_list_length - 1; i++)
    {
        state->loading_list[2 * i] = state->loading_list[2 * (i + 1)];
        state->loading_list[2 * i + 1] = state->loading_list[2 * (i + 1) + 1];
    }
    state->loading_list[2 * (state->loading_list_length - 1)] = 0;
    state->loading_list[2 * (state->loading_list_length - 1) + 1] = 0;
    state->loading_list_length -= 1;
}

// Right shifts the loading list (padding start with zeros)
void right_shift_loading_list(struct state *state)
{
    for (int i = state->loading_list_length; i > 0; i--)
    {
        state->loading_list[2 * i] = state->loading_list[2 * (i - 1)];
        state->loading_list[2 * i + 1] = state->loading_list[2 * (i - 1) + 1];
    }
    state->loading_list[0] = 0;
    state->loading_list[1] = 0;
    state->loading_list_length += 1;
}

int get_min_in_column(int j, struct state *state)
{
    // Initialize min_container to max container + 1
    int min_container = state->N;
    for (int i = state->R - 1; i >= 0; i--)
    {
        int bay_index = i * state->C + j;
        int container = state->bay_matrix[bay_index];
        // Reached top of stack
        if (container == 0)
        {
            break;
        }
        min_container = min(min_container, container);
    }

    return min_container;
}

// Offloads containers to the port, updates state and returns the number of shifts
int offload_containers(struct state *state)
{
    int n_shifts = 0;
    int *columns_to_min_check = get_zeros(state->C);
    for (int j = 0; j < state->C; j++)
    {
        int offloading_column = 0;
        for (int i = state->R - 1; i >= 0; i--)
        {
            int bay_index = i * state->C + j;
            int container = state->bay_matrix[bay_index];
            // We reached top of stack
            if (container == 0)
            {
                break;
            }
            if (container == state->port)
            {
                offloading_column = 1;
            }

            if (!offloading_column)
            {
                continue;
            }

            if (container != state->port)
            {
                // Add container back into transportation matrix
                int transportation_index = state->port * state->N + container;
                state->transportation_matrix[transportation_index] += 1;
                state->containers_per_port[state->port] += 1;
                n_shifts += 1;
            }

            if (container == state->min_container_per_column[j])
            {
                columns_to_min_check[j] = 1;
            }
            // Update state
            state->bay_matrix[bay_index] = 0;
            state->column_counts[j] -= 1;
        }
    }

    if (n_shifts > 0)
    {
        insert_loading_list(state);
    }

    for (int j = 0; j < state->C; j++)
    {
        if (columns_to_min_check[j])
        {
            state->min_container_per_column[j] = get_min_in_column(j, state);
        }
    }

    free(columns_to_min_check);

    return n_shifts;
}

// Adds container to bay and returns delta reward
int add_container(int i, int j, struct state *state)
{
    int delta_reward = 0;
    int container = state->loading_list[1];
    int bay_index = i * state->C + j;
    int transportation_index = state->port * state->N + container;

    // Update state
    state->bay_matrix[bay_index] = container;
    state->min_container_per_column[j] = min(state->min_container_per_column[j], container);
    state->column_counts[j] += 1;
    state->transportation_matrix[transportation_index] -= 1;
    state->containers_per_port[state->port] -= 1;

    // Either:
    // Remove the first container in the loading list
    // Or decrease the count of the first container in the loading list
    if (state->loading_list[0] == 1)
    {
        left_shift_loading_list(state);
    }
    else
    {
        state->loading_list[0] -= 1;
    }

    if (state->min_container_per_column[j] < container)
    {
        delta_reward -= 1;
    }

    // Sail along for every port that has no more containers to load
    while (state->containers_per_port[state->port] == 0)
    {
        state->port += 1;
        offload_containers(state);
        if (state->port + 1 == state->N)
        {
            break;
        }
    }

    return delta_reward;
}

void insert_transportation_matrix(struct state *state, int N, int R, int C, double exponential_constant, int seed)
{
    int capacity = R * C;
    srand(seed);
    for (int i = 0; i < N - 1; i++)
    {
        int *ordering = get_ordering(i + 1, N);
        int ordering_length = N - i - 1;
        for (int j = 0; j < ordering_length; j++)
        {
            int k = ordering[j];
            int matrix_index = i * N + k;
            if (j == ordering_length - 1)
            {
                // Make sure that ship is fully loaded
                state->transportation_matrix[matrix_index] = capacity;
                capacity = 0;
            }
            else
            {
                int random_value = (int)(ran_expo(exponential_constant));
                // Make sure that the ship is not overfilled
                state->transportation_matrix[matrix_index] = min(capacity, random_value);
                capacity -= state->transportation_matrix[matrix_index];
            }
            state->containers_per_port[i] += state->transportation_matrix[matrix_index];
        }
        free(ordering);

        // Offload containers
        for (int h = 0; h < i + 1; h++)
        {
            capacity += state->transportation_matrix[h * N + i + 1];
        }
    }
}

// Removes container from bay and returns delta reward
int remove_container(int i, int j, struct state *state)
{
    int bay_index = i * state->C + j;
    int container = state->bay_matrix[bay_index];

    // Add to beginning of loading list
    if (state->loading_list[1] == container)
    {
        state->loading_list[0] += 1;
    }
    else
    {
        right_shift_loading_list(state);
        state->loading_list[0] = 1;
        state->loading_list[1] = container;
    }

    // Update state
    state->bay_matrix[bay_index] = 0;
    state->column_counts[j] -= 1;
    state->transportation_matrix[state->port * state->N + container] += 1;
    state->containers_per_port[state->port] += 1;

    // Check if min_value_per_column needs to be checked/updated
    if (container == state->min_container_per_column[j])
    {
        state->min_container_per_column[j] = get_min_in_column(j, state);
    }

    return -1;
}

void free_state(struct state *state)
{
    free(state->bay_matrix);
    free(state->transportation_matrix);
    free(state->loading_list);
    free(state->column_counts);
    free(state->min_container_per_column);
    free(state->containers_per_port);
    free(state->mask);
    free(state);
}

struct state *get_state(int N, int R, int C, double exponential_constant, int seed)
{
    struct state *state = malloc(sizeof(struct state));
    state->N = N;
    state->R = R;
    state->C = C;
    state->port = 0;
    state->bay_matrix = get_zeros(R * C);
    state->column_counts = get_zeros(C);
    state->min_container_per_column = get_zeros(C);
    state->transportation_matrix = get_zeros(N * N);
    state->containers_per_port = get_zeros(N);
    state->mask = get_zeros(2 * C);
    int upper_triangle_length = state->N * (state->N - 1) / 2;
    // 2 * upper_triangle_length because we need to store the count and the container
    state->loading_list = get_zeros(2 * upper_triangle_length);
    state->loading_list_padded_length = upper_triangle_length;
    // Initialize min_container_per_column to N (max value + 1)
    for (int i = 0; i < C; i++)
    {
        state->min_container_per_column[i] = N;
    }
    insert_transportation_matrix(state, N, R, C, exponential_constant, seed);
    insert_loading_list(state);
    insert_mask(state);
    state->is_terminal = 0;
    state->last_reward = 0;
    state->last_action = -1;
    state->sum_reward = 0;
    return state;
}

// Execute one time step within the environment
void step(int action, struct state *state)
{
    // Assert that env is not terminal
    assert(!state->is_terminal);

    int should_add = action < state->C;
    int reward;
    if (should_add)
    {
        int j = action;
        // Placed top down
        int i = state->R - state->column_counts[j] - 1;
        // Cannot add containers to full columns
        assert(state->column_counts[j] < state->R);
        reward = add_container(i, j, state);
    }
    else
    {
        int j = action - state->C;
        int i = state->R - state->column_counts[j];
        // Cannot remove containers from empty columns
        assert(state->column_counts[j] > 0);
        reward = remove_container(i, j, state);
    }
    // Update mask
    insert_mask(state);

    state->is_terminal = state->port + 1 == state->N;
    state->last_reward = reward;
    state->last_action = action;
    state->sum_reward += reward;
}

// Returns a matrix of blocking containers (1 if blocking, 0 otherwise)
int *get_blocking(struct state *state)
{
    int *blocking_containers = get_zeros(state->R * state->C);
    for (int j = 0; j < state->C; j++)
    {
        int min_in_column = state->N;
        for (int i = state->R - 1; i >= 0; i--)
        {
            int bay_index = i * state->C + j;
            if (state->bay_matrix[bay_index] == 0)
            {
                break;
            }
            if (state->bay_matrix[bay_index] < min_in_column)
            {
                min_in_column = state->bay_matrix[bay_index];
            }
            if (state->bay_matrix[bay_index] > min_in_column)
            {
                blocking_containers[bay_index] = 1;
            }
        }
    }
    return blocking_containers;
}

void free_blocking(int *blocking_containers)
{
    free(blocking_containers);
}