#include <stdlib.h>
#include <stdio.h>

#include "ialib.h"

static void test_iterative()
{
    int states[] = { 0, 1, 2, 3, 4, 5,6,7,8,9};
    int nb_states = sizeof states / sizeof states[0];
    enum {ACTION_LEFT, ACTION_RIGHT};
    int actions[] = {ACTION_LEFT, ACTION_RIGHT};
    int nb_actions = sizeof actions / sizeof actions[0];
    double rewards[] = {-1, 0, 1};
    int nb_rewards = sizeof rewards / sizeof rewards[0];
    int etats_finaux[] = {0, nb_states-1};

    double (*transitions)[nb_states][nb_actions][nb_states][nb_rewards] = malloc(sizeof (double[nb_states][nb_actions][nb_states][nb_rewards]));

    for (int s = 1; s <= nb_states-3; s++) {
        (*transitions)[s][1][s+1][1] = 1.0;
    }
    
    for (int s = nb_states-3; s <= nb_states-2; s++) {
        (*transitions)[s][0][s-1][1] = 1.0;
    }

    (*transitions)[nb_states-2][1][nb_states-1][2] = 1.0;
    (*transitions)[1][0][0][0] = 1.0;

    double esperance[nb_states];
    for(int e = 0; e < nb_states ; e++) {
        if(e == 0 || e == nb_states -1) {
            esperance[e] = 0;
        } else {
            esperance[e] = get_rand_float();
        }
    }

    double theta = 0.0001;
    double gamma = 0.9999;

    {
        double policy_all_left[nb_states][nb_actions];
        for(int s = 0; s < nb_states ; s++) {
            policy_all_left[s][ACTION_LEFT] = 1;
            policy_all_left[s][ACTION_RIGHT] = 0;
        }

        ///test stratégie all left sur lineworld avec policy iterative
        ial_eval_policy_iterative(nb_states, nb_actions, policy_all_left, states, actions, nb_rewards, rewards, transitions, esperance, theta, gamma);
        print_esperance(esperance,nb_states);


    }

    {
        double policy_all_right[nb_states][nb_actions];
        for(int s = 0; s < nb_states ; s++) {
            policy_all_right[s][ACTION_LEFT] = 0;
            policy_all_right[s][ACTION_RIGHT] = 1;
        }

        //test stratégie all right sur lineworld avec policy iterative
        ial_eval_policy_iterative(nb_states, nb_actions, policy_all_right, states, actions, nb_rewards, rewards, transitions, esperance, theta, gamma);
        print_esperance(esperance,nb_states);

    }


    free(transitions);
}

void print_esperance(double *esperance,int nbstate){
    for (int s = 0 ; s<nbstate;s++){
        printf("%lf\t",esperance[s]);
    }
    printf("\n");
}

int main(void) 
{
    test_iterative();
}
