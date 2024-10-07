#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

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
    double pi[nb_states][nb_actions],
    int env_S[nb_states],
    int env_A[nb_actions],
    int nb_rewards,
    double env_R[nb_rewards],
    double (*env_probas)[nb_states][nb_actions][nb_states][nb_rewards],
    double esperance[nb_states],
    double theta,
    double gamma)
{
    while (1) {
        double delta = 0.0;

        for (int s = 0; s < nb_states; s++) {

            double v_prev = esperance[s];
            double total = 0.0;
            for (int a =0; a < nb_actions; a++) {
                for (int s_p = 0; s_p < nb_states; s_p++) {
                    for (int r_index = 0; r_index < nb_rewards; r_index++) {
                        double r = env_R[r_index];
                        total += pi[s][a] * (*env_probas)[s][a][s_p][r_index] * (r + gamma * esperance[s_p]);
                        //printf("%f\n",(*env_probas)[s][a][s_p][r_index]);
                    }
                }
            }

            esperance[s] = total;
            delta = max(delta, fabs(v_prev - esperance[s]));
        }
        
        if (delta < theta) {
            break;
        }
    }
}

