#ifndef ENV_HELPERS_H
#define ENV_HELPERS_H

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

// Defines what removals the agent is allowed
enum remove_restrictions
{
    remove_all = 0,
    remove_only_when_blocking = 1,
    no_remove = 2
};

// Contains information about the state of the environment
struct state
{
    int N;
    int R;
    int C;
    int port;
    int *bay_matrix;
    int *transportation_matrix;
    int *loading_list;
    int *column_counts;
    int *min_container_per_column;
    int *containers_per_port;
    int *mask;
    int loading_list_length;
    int loading_list_padded_length;
    int is_terminal;
    int last_reward;
    int last_action;
    int sum_reward;
    enum remove_restrictions remove_restrictions;
};

void step(int action, struct state *state);

struct state *get_state(int N, int R, int C, double exponential_constant, int seed, enum remove_restrictions remove_restrictions);

void free_state(struct state *state);

int *get_blocking(struct state *state);

void free_blocking(int *blocking_containers);

#endif