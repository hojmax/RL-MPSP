#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "env_helpers.h"
#include "hashmap.h"

struct QRow
{
    char *state;
    float *q_values;
};

void write_QRow(struct QRow *row, int n_actions, FILE *fp)
{
    for (int i = 0; i < n_actions; i++)
    {
        // 4 decimal precision
        fprintf(fp, "%.4f ", row->q_values[i]);
    }
    fprintf(fp, "%s\n", row->state);
}

void read_QRow(struct QRow *row, int n_actions, FILE *fp)
{
    for (int j = 0; j < n_actions; j++)
    {
        fscanf(fp, "%f ", &row->q_values[j]);
    }
    fscanf(fp, "%s\n", row->state);
}

int get_digit_count(int num)
{
    int count = 0;
    while (num != 0)
    {
        num /= 10;
        count++;
    }
    return count;
}

char *stateToString(struct state *state, int string_size)
{
    char *str = malloc(sizeof(char) * string_size);
    int index = 0;
    for (int i = 0; i < state->R; i++)
    {
        for (int j = 0; j < state->C; j++)
        {
            int value = state->bay_matrix[i * state->C + j];
            index += sprintf(&str[index], "%d|", value);
        }
    }
    // Only looping over upper triangle
    for (int i = 0; i < state->N - 1; i++)
    {
        for (int j = i + 1; j < state->N; j++)
        {
            int value = state->transportation_matrix[i * state->N + j];
            index += sprintf(&str[index], "%d|", value);
        }
    }

    return str;
}

int compare_QRow(const void *a, const void *b, void *udata)
{
    const struct QRow *rowA = a;
    const struct QRow *rowB = b;
    return strcmp(rowA->state, rowB->state);
}

uint64_t hash_QRow(const void *item, uint64_t seed0, uint64_t seed1)
{
    const struct QRow *row = item;
    return hashmap_sip(row->state, strlen(row->state), seed0, seed1);
}

void free_QRow(void *item)
{
    struct QRow *row = item;
    free(row->state);
    free(row->q_values);
    free(row);
}

int get_string_size(struct state *state)
{
    int capacity = state->R * state->C;
    int upper_triangle_size = state->N * (state->N - 1) / 2;
    // The bay matrix is encoded as:
    // 9|11|1|0|...
    // In this example ^, N=12, meaning the last port is 11
    // We therefore need to take into account the amount of digits
    // in the last port number, to find an upper bound on string size
    // +1 for the vertical pipe |
    int bay_matrix_size = capacity * (get_digit_count(state->N - 1) + 1);
    int transportation_matrix_size = upper_triangle_size * (get_digit_count(capacity) + 1);
    int string_size = bay_matrix_size + transportation_matrix_size;
    return string_size;
}

float *get_default_q_values(struct state *state)
{
    float *q_values = malloc(sizeof(int) * 2 * state->C);
    for (int i = 0; i < 2 * state->C; i++)
    {
        q_values[i] = i < state->C ? 0 : -1;
    }
    return q_values;
}

int main()
{
    int N = 4;
    int R = 6;
    int C = 2;
    double exponential_constant = 0.25;
    int seed = 0;

    struct state *state = get_empty_state(N, R, C, remove_all);
    initialize_random_state(state, exponential_constant, seed);

    int string_size = get_string_size(state);
    char *state_str = stateToString(state, string_size);
    printf("%s\n", state_str);

    struct hashmap *map = hashmap_new(sizeof(struct QRow), 0, 0, 0,
                                      hash_QRow, compare_QRow, NULL, NULL);

    struct QRow *row = malloc(sizeof(struct QRow));
    row->state = state_str;
    row->q_values = get_default_q_values(state);
    hashmap_set(map, row);
    free_QRow(row);

    struct QRow *row2;
    row2 = hashmap_get(map, &(struct QRow){.state = state_str});
    printf("State: %s\n", row2->state);

    for (int i = 0; i < 2 * C; i++)
    {
        printf("%.4f ", row2->q_values[i]);
    }

    hashmap_free(map);
    free_state(state);
}