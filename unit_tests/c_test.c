#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "env_helpers.h"

int test_exponential(double exponential_constant, int iterations)
{
    int total_rewards = 0;
    struct state *state = get_empty_state(
        10,       // N
        10,       // R
        4,        // C
        no_remove // Only add
    );
    // Seeding rand
    srand(0);
    for (int i = 0; i < iterations; i++)
    {
        initialize_random_state(state, exponential_constant, i); // Seed is i
        while (!state->is_terminal)
        {
            int *possible = malloc(sizeof(int) * state->C);
            int possible_index = 0;
            for (int i = 0; i < state->C; i++)
            {
                if (state->mask[i] == 1)
                {
                    possible[possible_index] = i;
                    possible_index++;
                }
            }
            assert(possible_index > 0);
            // Choose possible random action
            int action = possible[rand() % possible_index];
            assert(state->mask[action] == 1);
            step(action, state);
            free(possible);
        }
        total_rewards += state->sum_reward;
    }
    free_state(state);
    // Print average reward to 2 decimals
    printf("Average reward: %.2f ", (double)total_rewards / iterations);
    printf("Expontential constant: %f\n", exponential_constant);
}

int main()
{
    int iterations = 50000;
    printf("Iterations: %d\n", iterations);
    for (int i = 17; i <= 24; i++)
    {
        test_exponential((double)i / 100, iterations);
    }
}

// Command to compile:
// gcc -o c_test c_test.c env_helpers.c -lm
// Run the command with leaks to check for memory leaks:
// leaks -atExit -- ./c_test