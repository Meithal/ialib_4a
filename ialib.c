#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include "ialib.h"


char* ial_version(void)
{
    return "0.0.1";
}

/**
 * Returns a int between 0 et RAND_MAX
 */
int random_int() {
    static int srand_called;

    if(!srand_called) {
        srand(time(NULL));
        srand_called = 1;
    }

    return rand();
}

_Bool ial_same(double a, double b) {
    return fabs(a - b) < DBL_EPSILON;
}


void update_policy_probability(
        int nb_actions, double(* policy)[nb_actions],
        int action, double value) {
    assert(value >=0. && value <= 1.);

    double difference = value - (*policy)[action];
    double to_remove = difference / (nb_actions - 1);
    for (int i = 0; i < nb_actions; ++i) {
        if(i == action) {
            (*policy)[i] = value;
        } else {
            (*policy)[i] = (*policy)[i] - to_remove;
        }
    }
}


/**
 * Given a policy, calculates the value of each state if we keep following that policy
 */
void ial_eval_policy_iterative(
    int nb_states,
    int nb_actions,
    double pi[nb_states][nb_actions],
    int nb_rewards,
    const double env_R[nb_rewards],
    double state_value_for_policy[nb_states],
    double theta,
    double gamma,
    struct ial_env* env)
{
    double (*env_probas)[nb_states][nb_actions][nb_states][nb_rewards] = env->_probabs_transitions;

    while (1) {
        double delta = 0.0;

        for (int s = 0; s < nb_states; s++) {

            double v_prev = state_value_for_policy[s];
            double total = 0.0;
            for (int a =0; a < nb_actions; a++) {
                for (int s_p = 0; s_p < nb_states; s_p++) {
                    for (int r_index = 0; r_index < nb_rewards; r_index++) {
                        double r = env_R[r_index];
                        total += pi[s][a] * ((*env_probas)[s][a][s_p][r_index] * (r + gamma * state_value_for_policy[s_p]));
                    }
                }
            }

            state_value_for_policy[s] = total;
            delta = fmax(delta, fabs(v_prev - state_value_for_policy[s]));
        }
        
        if (delta < theta) {
            break;
        }
    }
}

void ial_eval_policy_iterative_fun(
    const int nb_states,
    const int nb_actions,
    const double pi[nb_states][nb_actions],
    const double (* const f_rew)(const int nb_states, const int state, const int action, int* out_state),
    const double theta,
    const double gamma,
    double state_value_for_policy[nb_states]
    )
{
    while (1) {
        double delta = 0.0;

        for (int s = 0; s < nb_states; s++) {

            double v_prev = state_value_for_policy[s];
            double total = 0.0;
            for (int a =0; a < nb_actions; a++) {
                int s_p;
                double r = f_rew(nb_states, s, a, &s_p);
                total += pi[s][a] * (r + gamma * state_value_for_policy[s_p]);
                //printf("%f\n",(*env_probas)[s][a][s_p][r_index]);
            }

            state_value_for_policy[s] = total;
            delta = fmax(delta, fabs(v_prev - state_value_for_policy[s]));
        }
        
        if (delta < theta) {
            break;
        }
    }
}

/*
void ial_policy_iteration(
    int nb_states,
    int nb_actions,
    int nb_rewards,
    int env_A[nb_actions],
    double env_R[nb_rewards],
    double (*env_probas)[nb_states][nb_actions][nb_states][nb_rewards],
    double state_value_for_policy[nb_states],
    double theta,
    double gamma)
{
    enum {ACTION_LEFT, ACTION_RIGHT};
    double policy_random[nb_states][nb_actions];
    for(int s = 0; s < nb_states ; s++) {
        srand(time(NULL));
        float randomFloat = rand() / (float)RAND_MAX;// Générer un nombre flottant aléatoire entre 0 et 1
        policy_random[s][ACTION_LEFT] = randomFloat;
        policy_random[s][ACTION_RIGHT] = 1-randomFloat;
    }

    while(1){
        ial_eval_policy_iterative(nb_states, nb_actions, policy_random, nb_rewards, env_R, state_value_for_policy, theta, gamma, );
        int policy_stable = 1;

        for(int s = 0;s<nb_states;s++){
            double **old_action = policy_random; //?????
            //double new_p = max(policy_random[s][ACTION_LEFT,])
        }
    }
}
*/

void ial_policy_iteration_naive(
        const int nb_states,
        const int nb_actions,
        const double (* f_rew)(const int nb_states, const int state, const int action, int *out_state),
        const double theta,
        const double gamma,
        double pi[nb_states][nb_actions],
        double state_value_for_policy[nb_states]
)
{
    while (1) {
        double delta = 0.0;

        for (int s = 0; s < nb_states; s++) {

            double v_prev = state_value_for_policy[s];
            double best[nb_actions];
            double total = 0;
            for (int a =0; a < nb_actions; a++) {
                int s_p;
                double r = f_rew(nb_states, s, a, &s_p);
                best[a] = 1 * (r + gamma * state_value_for_policy[s_p]);
                total += 1 * (r + gamma * state_value_for_policy[s_p]);
            }

            for (int a = 0; a < nb_actions; ++a) {
                pi[s][a] = best[a];
            }

            state_value_for_policy[s] = total;
            delta = fmax(delta, fabs(v_prev - state_value_for_policy[s]));
        }

        if (delta < theta) {
            break;
        }
    }
}

/**
 * Here, we have for each state an array of possible actions and their value if we follow a given policy
 */
void q_learning(
        int nb_states,
        int nb_actions,
        float (*q_learning)[nb_states][nb_actions],
        int nb_episodes,
        float learning_rate,
        float gamma,
        float epsilon

) {

}