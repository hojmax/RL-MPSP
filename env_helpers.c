#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

// Contains information about the state of the environment
struct state
{
    int *transportation_matrix;
    int *bay_matrix;
    int *loading_list;
    int loading_list_length;
    int loading_list_padded_length;
    int *column_counts;
    int *min_container_per_column;
    int *containers_per_port;
    int N;
    int R;
    int C;
    int port;
};

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

// Generates the loading list and inserts it into the state
void insert_loading_list(struct state *state)
{
    int upper_triangle_length = state->N * (state->N - 1) / 2;
    int list_index = 0;
    // 2 * upper_triangle_length because we need to store the count and the container
    int *list = get_zeros(2 * upper_triangle_length);

    // Iterate over the upper triangle of the transportation matrix
    for (int i = 0; i < state->N - 1; i++)
    {
        for (int j = state->N - 1; j >= i + 1; j--)
        {
            int matrix_index = i * state->N + j;
            int count = state->transportation_matrix[matrix_index];
            if (count > 0)
            {
                int container = j;
                list[2 * list_index] = count;
                list[2 * list_index + 1] = container;
                list_index++;
            }
        }
    }

    state->loading_list = list;
    state->loading_list_length = list_index;
    state->loading_list_padded_length = upper_triangle_length;
}

// Left shifts the loading list (removes first element)
int left_shift_loading_list(struct state *state)
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

// Right shifts the loading list (zeroes first element)
int right_shift_loading_list(struct state *state, int offset)
{
    for (int i = state->loading_list_length; i > offset; i--)
    {
        state->loading_list[2 * i] = state->loading_list[2 * (i - 1)];
        state->loading_list[2 * i + 1] = state->loading_list[2 * (i - 1) + 1];
    }
    state->loading_list[offset] = 0;
    state->loading_list[offset + 1] = 0;
    state->loading_list_length += 1;
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

// Offloads containers to the port, updates state and returns the number of shifts
void offload_containers(struct state *state)
{
    // TODO: Should check min_container_per_column
    int *readditions = get_zeros(state->N);
    int n_shifts = 0;
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
                // Readdition
                readditions[container] += 1;
                n_shifts += 1;
            }

            // Update state
            state->bay_matrix[bay_index] = 0;
            state->column_counts[j] -= 1;
        }
    }
    int loading_list_index = 0;
    int transportation_matrix_index = state->N - 1;
    // Add shifts back into the loading list and transportation matrix
    // for (int i = state->N - 1; i >= 0; i--)
    // {
    //     if (readditions[i] > 0)
    //     {
    //         while (i < transportation_matrix_index)
    //         {
    //             int k = state->N * state->port + transportation_matrix_index;
    //             if (state->transportation_matrix[k] > 0)
    //             {
    //                 loading_list_index++;
    //             }
    //             transportation_matrix_index -= 1;
    //         }
    //         int k = state->N * state->port + transportation_matrix_index;
    //         if (state->transportation_matrix[k] > 0)
    //         {
    //             state->loading_list[2 * loading_list_index] += readditions[i];
    //         }
    //         else
    //         {
    //             right_shift_loading_list(state, loading_list_index);
    //             state->loading_list[2 * loading_list_index] = readditions[i];
    //             state->loading_list[2 * loading_list_index + 1] = i;
    //             loading_list_index++;
    //         }
    //         state->transportation_matrix[k] += readditions[i];
    //     }
    // }
    return n_shifts;
}

void insert_transportation_matrix(struct state *state, int N, int R, int C, double exponential_constant, int seed)
{
    int *matrix = get_zeros(N * N);
    int *containers_per_port = get_zeros(N);
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
                matrix[matrix_index] = capacity;
                capacity = 0;
            }
            else
            {
                int random_value = (int)(ran_expo(exponential_constant));
                // Make sure that the ship is not overfilled
                matrix[matrix_index] = min(capacity, random_value);
                capacity -= matrix[matrix_index];
            }
            containers_per_port[i] += matrix[matrix_index];
        }
        free(ordering);

        // Offload containers
        for (int h = 0; h < i + 1; h++)
        {
            capacity += matrix[h * N + i + 1];
        }
    }

    return matrix;
}

void free_array(int *array)
{
    free(array);
}