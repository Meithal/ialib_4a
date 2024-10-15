#include <stdlib.h>
#include <math.h>

#include "ialib.h"


char* ial_version(void)
{
    return "0.0.1";
}

/**
 * Retourne un int entre 0 et RAND_MAX
 */
int random_int() {
    static int srand_called;

    if(!srand_called) {
        srand(time(NULL));
        srand_called = 1;
    }

    return rand();
}

/**
 * Retourne un flottant entre 0 et 1
 */
double rand_float() {
    return random_int() / (double)RAND_MAX;
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
    int nb_rewards,
    const double env_R[nb_rewards],
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
                        total += pi[s][a] * ((*env_probas)[s][a][s_p][r_index] * (r + gamma * esperance[s_p]));
                        //printf("%f\n",(*env_probas)[s][a][s_p][r_index]);
                    }
                }
            }

            esperance[s] = total;
            printf("%f\n",delta);
            delta = max(delta, abs(v_prev - esperance[s]));
            printf("%f\n",delta);
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
    const double (* const f_rew)(int state, int action, int* out_state),
    const double theta,
    const double gamma,
    double esperance[nb_states]
    )
{
    while (1) {
        double delta = 0.0;

        for (int s = 0; s < nb_states; s++) {

            double v_prev = esperance[s];
            double total = 0.0;
            for (int a =0; a < nb_actions; a++) {
                int s_p;
                double r = f_rew(s, a, &s_p);
                total += pi[s][a] * (r + gamma * esperance[s_p]);
                //printf("%f\n",(*env_probas)[s][a][s_p][r_index]);
            }

            esperance[s] = total;
            delta = max(delta, fabs(v_prev - esperance[s]));
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
    double esperance[nb_states],
    double theta,
    double gamma)
{
    enum {ACTION_LEFT, ACTION_RIGHT};
    double policy_random[nb_states][nb_actions];
    for(int s = 0; s < nb_states ; s++) {
        srand(time(NULL));
        float randomFloat = (float)rand() / (float)RAND_MAX;// Générer un nombre flottant aléatoire entre 0 et 1
        policy_random[s][ACTION_LEFT] = randomFloat;
        policy_random[s][ACTION_RIGHT] = 1-randomFloat;
    }

    while(1){
        ial_eval_policy_iterative(nb_states, nb_actions, policy_random, env_S, env_A, nb_rewards, env_R, env_probas, esperance, theta, gamma);
        bool policy_stable = true;

        for(int s = 0;s<nb_states;s++){
            double **old_action = policy_random; //?????
            //double new_p = max(policy_random[s][ACTION_LEFT,])
        } 
    }

    




}
*/

