#include <stdlib.h>

#include "ialib.h"

static void test_iterative()
{
    int states[] = { 0, 1, 2, 3, 4 };
    int nb_states = sizeof states / sizeof states[0];
    enum {ACTION_LEFT, ACTION_RIGHT};
    int actions[] = {ACTION_LEFT, ACTION_RIGHT};
    int nb_actions = sizeof actions / sizeof actions[0];
    float rewards[] = {-1, 0, 1};
    int nb_rewards = sizeof rewards / sizeof rewards[0];
    int etats_finaux[] = {0, 4};

    float (*transitions)[nb_states][nb_actions][nb_states][nb_rewards] = malloc(sizeof (float[nb_states][nb_actions][nb_states][nb_rewards]));


    float esperance[nb_states];
    for(int e = 0; e < nb_states ; e++) {
        if(e == 0 || e == nb_states -1) {
            esperance[e] = 0;
        } else {
            esperance[e] = get_rand_float();
        }
    }

    {
        float policy_all_left[nb_states][nb_actions];
        for(int s = 0; s < nb_states ; s++) {
            policy_all_left[s][ACTION_LEFT] = 1;
            policy_all_left[s][ACTION_RIGHT] = 0;
        }

        float theta = 0.0001;
        float gamma = 0.9999;

        ial_eval_policy_iterative(nb_states, nb_actions, policy_all_left, states, actions, nb_rewards, rewards, transitions, esperance, theta, gamma);
    }

    free(transitions);
}

int main(void) 
{
    test_iterative();
}
