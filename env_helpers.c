#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

struct transportation_info
{
    int *loading_list;
    int *transportation_matrix;
    int loading_list_length;
};

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

struct transportation_info _get_transportation_info(int N, int R, int C, double exponential_constant, double exponential_multiplier, int seed)
{
    struct transportation_info info;
    int *matrix = malloc(N * N * sizeof(int));
    int upper_triangle_length = N * (N - 1) / 2;
    // 2 * upper_triangle_length because we need to store the count and the container
    int *list = malloc(2 * upper_triangle_length * sizeof(int));
    int list_index = 0;
    int capacity = R * C;
    for (int i = 0; i < N * N; i++)
    {
        matrix[i] = 0;
    }
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
                int random_value = (int)(ran_expo(exponential_constant) * exponential_multiplier);
                matrix[matrix_index] = min(capacity, random_value);
                capacity -= matrix[matrix_index];
            }
            if (matrix[matrix_index] > 0)
            {
                // Count
                list[2 * list_index] = matrix[matrix_index];
                // Container
                list[2 * list_index + 1] = k;
                list_index++;
            }
        }
        free(ordering);

        for (int h = 0; h < i + 1; h++)
        {
            capacity += matrix[h * N + i + 1];
        }
    }
    info.transportation_matrix = matrix;
    info.loading_list = list;
    info.loading_list_length = list_index;

    return info;
}

void free_transportation_info(struct transportation_info info)
{
    free(info.transportation_matrix);
    free(info.loading_list);
}