#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "env_helpers.h"
#include <assert.h>

// Fills an array with a value
void fill_array(int *array, int length, int value)
{
    for (int i = 0; i < length; i++)
    {
        array[i] = value;
    }
}

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
void update_mask_for_column(struct state *state, int j)
{
    // Add mask
    state->mask[j] = state->column_counts[j] < state->R;

    // Remove mask
    if (state->remove_restrictions == remove_all)
    {
        int column_not_empty = state->column_counts[j] > 0;
        state->mask[j + state->C] = column_not_empty;
    }
    else if (state->remove_restrictions == remove_only_when_blocking)
    {
        int column_not_empty = state->column_counts[j] > 0;
        int next_container = state->loading_list[1];
        int is_blocking = state->min_container_per_column[j] < next_container;
        state->mask[j + state->C] = column_not_empty && is_blocking;
    }
    else if (state->remove_restrictions == no_remove)
    {
        state->mask[j + state->C] = 0;
    }
}

void insert_mask(struct state *state)
{
    for (int j = 0; j < state->C; j++)
    {
        update_mask_for_column(state, j);
    }
}

// Subtract from every element in the bay_matrix that is not 0
void decrement_bay_matrix(struct state *state, int delta_port)
{
    for (int i = 0; i < state->R * state->C; i++)
    {
        if (state->bay_matrix[i] != 0)
        {
            state->bay_matrix[i] -= delta_port;
        }
    }
}

void decrement_min_container_per_column(struct state *state, int delta_port)
{
    for (int j = 0; j < state->C; j++)
    {
        // min_container_per_column is N when the column is empty
        if (state->min_container_per_column[j] < state->N)
        {
            state->min_container_per_column[j] -= delta_port;
        }
    }
}

void reshift_transportation_matrix(struct state *state, int delta_port)
{
    for (int i = 0; i < state->N - 1 - delta_port; i++)
    {
        for (int j = i + 1; j < state->N - delta_port; j++)
        {
            state->transportation_matrix[i * state->N + j] = state->transportation_matrix[(i + delta_port) * state->N + (j + delta_port)];
            state->transportation_matrix[(i + delta_port) * state->N + (j + delta_port)] = 0;
        }
    }
}

void reshift_containers_per_port(struct state *state, int delta_port)
{
    for (int i = 0; i < state->N - delta_port; i++)
    {
        state->containers_per_port[i] = state->containers_per_port[i + delta_port];
        state->containers_per_port[i + delta_port] = 0;
    }
}

void decrement_loading_list(struct state *state, int delta_port)
{
    for (int i = 0; i < state->loading_list_length; i++)
    {
        if (state->loading_list[2 * i + 1] == 0)
        {
            break;
        }
        state->loading_list[2 * i + 1] -= delta_port;
    }
}

// Reshift transportation_matrix, and decrement bay_matrix and min_container_per_column
void zero_port_normalization(struct state *state, int delta_port)
{
    decrement_bay_matrix(state, delta_port);
    decrement_min_container_per_column(state, delta_port);
    reshift_transportation_matrix(state, delta_port);
    reshift_containers_per_port(state, delta_port);
    decrement_loading_list(state, delta_port);
    state->port = 0;
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
                state->loading_list[2 * list_index + 1] = j; // j is container
                list_index++;
            }
        }
    }

    state->loading_list = state->loading_list;
    state->loading_list_length = list_index;
}

// Sort using count sort
// With n-1 being the maximum value in the array
void sort_according_to_values(int *values, int *array, int length, int max_value)
{
    int *count = calloc(max_value, sizeof(int));
    for (int i = 0; i < length; i++)
    {
        count[values[i]]++;
    }

    for (int i = 1; i < max_value; i++)
    {
        count[i] += count[i - 1];
    }

    int *output = malloc(length * sizeof(int));
    for (int i = length - 1; i >= 0; i--)
    {
        output[count[values[i]] - 1] = array[i];
        count[values[i]]--;
    }

    for (int i = 0; i < length; i++)
    {
        array[i] = output[i];
    }
    free(count);
    free(output);
}

// Takes a R x C array
// Values must be in the range [0, N-1]
// Alters (in place) the array with the columns swapped in order
// The order is lexographical (bottom being most significant)
// Example:
// Input
// 0 0 0 1
// 0 1 0 1
// 2 1 0 1
// 1 2 3 1
// Output
// 1 0 0 0
// 1 0 1 0
// 1 2 1 0
// 1 1 2 3
void sort_bay_columns(struct state *state)
{
    int *column_order = malloc(state->C * sizeof(int));
    // Starts with 0 to n-1, as the current order
    for (int i = 0; i < state->C; i++)
    {
        column_order[i] = i;
    }

    for (int i = 0; i < state->R; i++)
    {
        int some_non_zero_value = 0;
        int *row_values = malloc(state->C * sizeof(int));
        for (int j = 0; j < state->C; j++)
        {
            int value = state->bay_matrix[i * state->C + column_order[j]];
            row_values[j] = value;
            if (value > 0)
            {
                some_non_zero_value = 1;
            }
        }
        // No need to sort if all values are zero
        if (some_non_zero_value)
        {
            sort_according_to_values(row_values, column_order, state->C, state->N);
        }
        free(row_values);
    }

    // Swap colums into new order, in place
    for (int i = 0; i < state->R; i++)
    {
        int *row = malloc(state->C * sizeof(int));
        for (int j = 0; j < state->C; j++)
        {
            row[j] = state->bay_matrix[i * state->C + j];
        }
        for (int j = 0; j < state->C; j++)
        {
            state->bay_matrix[i * state->C + j] = row[column_order[j]];
        }
        free(row);
    }

    int *prev_column_counts = malloc(state->C * sizeof(int));
    int *prev_min_container_per_column = malloc(state->C * sizeof(int));
    int *prev_mask = malloc(2 * state->C * sizeof(int));
    for (int j = 0; j < state->C; j++)
    {
        prev_column_counts[j] = state->column_counts[j];
        prev_min_container_per_column[j] = state->min_container_per_column[j];
    }
    for (int j = 0; j < 2 * state->C; j++)
    {
        prev_mask[j] = state->mask[j];
    }
    for (int j = 0; j < state->C; j++)
    {
        // Swap the column counts
        state->column_counts[j] = prev_column_counts[column_order[j]];
        // Swap the min container per column
        state->min_container_per_column[j] = prev_min_container_per_column[column_order[j]];
        // Swap the add-part of the mask
        state->mask[j] = prev_mask[column_order[j]];
        // Swap the remove-part of the mask
        state->mask[state->C + j] = prev_mask[state->C + column_order[j]];
    }
    free(prev_column_counts);
    free(prev_min_container_per_column);
    free(prev_mask);
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

// Get minimum container in column j
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

    insert_mask(state);

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

    int prev_port = state->port;

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

    int delta_port = state->port - prev_port;

    if (delta_port > 0)
    {
        zero_port_normalization(state, delta_port);
    }
    else
    {
        // Update mask
        // Mask is already handled in offload_containers, so only check when not changing port
        update_mask_for_column(state, j);
    }
    return delta_reward;
}

void insert_transportation_matrix(struct state *state, double exponential_constant, int seed)
{
    int capacity = state->R * state->C;
    srand(seed);

    for (int i = 0; i < state->N - 1; i++)
    {
        int *ordering = get_ordering(i + 1, state->N);
        int ordering_length = state->N - i - 1;
        for (int j = 0; j < ordering_length; j++)
        {
            // Nothing more to add to row
            if (capacity == 0)
            {
                break;
            }
            int k = ordering[j];
            int matrix_index = i * state->N + k;
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
            capacity += state->transportation_matrix[h * state->N + i + 1];
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

    update_mask_for_column(state, j);

    return -1;
}

// Frees the memory of a state
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

struct state *get_empty_state(int N, int R, int C, enum remove_restrictions remove_restrictions)
{
    assert(N > 0);
    assert(R > 0);
    assert(C > 0);
    struct state *state = malloc(sizeof(struct state));
    int upper_triangle_length = N * (N - 1) / 2;
    state->N = N;
    state->R = R;
    state->C = C;
    state->bay_matrix = malloc(R * C * sizeof(int));
    state->transportation_matrix = malloc(N * N * sizeof(int));
    // 2 * upper_triangle_length because we need to store the count and the container
    state->loading_list = malloc(2 * upper_triangle_length * sizeof(int));
    state->column_counts = malloc(C * sizeof(int));
    state->min_container_per_column = malloc(C * sizeof(int));
    state->containers_per_port = malloc(N * sizeof(int));
    state->mask = malloc(2 * C * sizeof(int));
    state->remove_restrictions = remove_restrictions;
    state->loading_list_padded_length = upper_triangle_length;
    return state;
}

// Clears the state
void clear_state(struct state *state)
{
    fill_array(state->bay_matrix, state->R * state->C, 0);
    fill_array(state->transportation_matrix, state->N * state->N, 0);
    fill_array(state->loading_list, 2 * state->loading_list_padded_length, 0);
    fill_array(state->column_counts, state->C, 0);
    // Initialize min_container_per_column to N (max value + 1)
    fill_array(state->min_container_per_column, state->C, state->N);
    fill_array(state->containers_per_port, state->N, 0);
    fill_array(state->mask, 2 * state->C, 0);
    state->port = 0;
    state->is_terminal = 0;
    state->last_reward = 0;
    state->last_action = -1;
    state->sum_reward = 0;
}

void initialize_random_state(struct state *state, double exponential_constant, int seed)
{
    clear_state(state);
    insert_transportation_matrix(state, exponential_constant, seed);
    insert_loading_list(state);
    insert_mask(state);
}

void insert_containers_per_port(struct state *state)
{
    for (int i = 0; i < state->N - 1; i++)
    {
        for (int j = i + 1; j < state->N; j++)
        {
            state->containers_per_port[i] += state->transportation_matrix[i * state->N + j];
        }
    }
}

void initialize_state_from_transportation_matrix(struct state *state, int *transportation_matrix)
{
    clear_state(state);
    state->transportation_matrix = transportation_matrix;
    insert_containers_per_port(state);
    insert_loading_list(state);
    insert_mask(state);
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
    sort_bay_columns(state);

    state->is_terminal = state->loading_list_length == 0;
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