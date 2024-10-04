#include <stdlib.h>
#include <time.h>

char* ial_version(void)
{
    return "0.0.1";
}

/**
 * Retourne une int entre 0 et RAND_MAX
 */
int get_random_int() {
    static int srand_called;

    if(!srand_called) {
        srand(time(NULL));
        srand_called = 1;
    }

    return rand();
}

/**
 * Retourne un floattant entre 0 et 1
 */
double get_rand_float() {
    return get_random_int() / (double)RAND_MAX;
}

double max(double a, double b) {
    return a > b ? a : b;
}

double min(double a, double b) {
    return a < b ? a : b;
}


void ial_eval_policy_iterative(
    int nb_states,
    int nb_actions,
    float pi[nb_states][nb_actions],
    float env_S[nb_states],
    float env_A[nb_actions],
    int nb_rewards,
    int env_R[nb_rewards],
    float env_probas[nb_states][nb_actions][nb_states][nb_rewards],
    float esperance[nb_states],
    float theta,
    float gamma)
{
    while (1) {
        float delta = 0.0;
        for (int s = 0; s < nb_states; s++) {
            float v_prev = esperance[s];
            float total = 0.0;
            for (int a =0; a < nb_actions; a++) {
                for (int s_p = 0; s_p < nb_states; s_p++) {
                    for (int r_index = 0; r_index < nb_rewards; r_index++) {
                        int r = env_R[r_index];
                        total += pi[s][a] * env_probas[s][a][s_p][r_index] * (r + gamma * esperance[s_p]);
                    }
                }
            }
            esperance[s] = total;
            delta = max(delta, abs(v_prev - esperance[s]));
        }
        if (delta < theta) {
            break;
        }
    }
}

