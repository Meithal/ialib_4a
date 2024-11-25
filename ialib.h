#include <time.h>

int random_int();

static inline long get_nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_nsec;
}

_Bool ial_same(double a, double b);

void print_esperance(double *esperance,int nbstate);


char* ial_version(void);

struct ial_env {
    int _nb_states;
    int _nb_actions;
    int _nb_rewards;
    void (* _probabs_transitions);
};

static inline int ialenv_get_nb_states(struct ial_env env) {
    return env._nb_states;
}

static inline int ialenv_get_nb_actions(struct ial_env env) {
    return env._nb_actions;
}

static inline int ialenv_get_nb_rewards(struct ial_env env) {
    return env._nb_rewards;
}

/**
 * for a state in a policy, set a value and update the other
 * ones so the sum remains 1.
 */
void update_policy_probability(int nb_actions, double(*)[nb_actions],
                                  int action, double value);

void ial_eval_policy_iterative(
    int nb_states,
    int nb_actions,
    double pi[nb_states][nb_actions],
    int nb_rewards,
    const double env_R[nb_rewards],
    double state_value_for_policy[nb_states],
    double theta,
    double gamma,
    struct ial_env* env
    );

void ial_eval_policy_iterative_fun(
    const int nb_states,
    const int nb_actions,
    const double pi[nb_states][nb_actions],
    const double (* const f_rew)(const int nb_states, const int state, const int action, int *out_state),
    const double theta,
    const double gamma,
    double state_value_for_policy[nb_states]
    );

void ial_policy_iteration(
    int nb_states,
    int nb_actions,
    int nb_rewards,
    int env_A[nb_actions],
    double env_R[nb_rewards],
    double (*env_probas)[nb_states][nb_actions][nb_states][nb_rewards],
    double esperance[nb_states],
    double theta,
    double gamma);

void ial_policy_iteration_naive(
        int nb_states,
        int nb_actions,
        const double (* f_rew)(const int nb_states, const int state, const int action, int *out_state),
        double theta,
        double gamma,
        double pi[nb_states][nb_actions],
        double state_value_for_policy[nb_states]
);

void q_learning(
        int nb_states,
        int nb_actions,
        float (*q_learning)[nb_states][nb_actions],
        int nb_episodes,
        float learning_rate,
        float gamme,
        float epsilon

        );