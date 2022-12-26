#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

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

int *_get_mixed_transportation_matrix(int N, int R, int C, double exponential_constant, double exponential_multiplier, int seed)
{
    int *matrix = malloc(N * N * sizeof(int));
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
            int random_value = (int)(ran_expo(exponential_constant) * exponential_multiplier);
            matrix[i * N + k] = min(capacity, random_value);
            capacity -= matrix[i * N + k];
        }
        free(ordering);
        if (capacity > 0)
        {
            // Make sure that ship is fully loaded
            int random_index = random_int(i + 1, N);
            matrix[i * N + random_index] = capacity;
            capacity = 0;
        }

        for (int h = 0; h < i + 1; h++)
        {
            capacity += matrix[h * N + i + 1];
        }
    }
    return matrix;
}