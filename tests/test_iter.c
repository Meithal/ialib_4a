#include <stdlib.h>
#include <stdio.h>


#include "ialib.h"

enum actions {ACTION_LEFT, ACTION_RIGHT};
enum rewards {REW_LOSS, REW_NEUTRAL, REW_WIN};

enum {nb_states = 10};
int actions[] = {ACTION_LEFT, ACTION_RIGHT};
enum {nb_actions = sizeof actions / sizeof actions[0]};
double rewards[] = {[REW_LOSS] = -1, [REW_NEUTRAL] = 0, [REW_WIN]=1};
enum {nb_rewards = sizeof rewards / sizeof rewards[0]};

static void test_iterative()
{
    double (*transitions)[nb_states][nb_actions][nb_states][nb_rewards] = malloc(sizeof (double[nb_states][nb_actions][nb_states][nb_rewards]));

    for (int s = 1; s <= nb_states-3; s++) {
        (*transitions)[s][ACTION_RIGHT][s+1][REW_NEUTRAL] = 1.0;
    }
    
    for (int s = 2; s <= nb_states-2; s++) {
        (*transitions)[s][ACTION_LEFT][s-1][REW_NEUTRAL] = 1.0;
    }

    (*transitions)[nb_states-2][ACTION_RIGHT][nb_states-1][REW_WIN] = 1.0;
    (*transitions)[1][ACTION_LEFT][0][REW_LOSS] = 1.0;

    double esperance[nb_states] = {0};

    double theta = 0.0001;
    double gamma = 0.9999;

    {
        double policy_all_left[nb_states][nb_actions];
        for(int s = 0; s < nb_states ; s++) {
            policy_all_left[s][ACTION_LEFT] = 1;
            policy_all_left[s][ACTION_RIGHT] = 0;
        }

        ///test stratégie all left sur lineworld avec policy iterative
        ial_eval_policy_iterative(
                nb_states, nb_actions, policy_all_left, nb_rewards, rewards,
                transitions, esperance, theta, gamma);
        print_esperance(esperance,nb_states);
    }

    {
        double policy_all_right[nb_states][nb_actions];
        for(int s = 0; s < nb_states ; s++) {
            policy_all_right[s][ACTION_LEFT] = 0;
            policy_all_right[s][ACTION_RIGHT] = 1;
        }

        //test stratégie all right sur lineworld avec policy iterative
        ial_eval_policy_iterative(nb_states, nb_actions, policy_all_right, nb_rewards,
                                  rewards, transitions, esperance, theta, gamma);
        print_esperance(esperance,nb_states);
    }

    free(transitions);
}

const double reward(int state, int action, int* out_state)
{
    double rew;
    if(action == ACTION_LEFT) {
        *out_state = state>0 ? state-1: 0;

        if(state == 1) {
            rew = rewards[REW_LOSS];
        } else {
            rew = rewards[REW_NEUTRAL];
        }
    } else { /// (action == ACTION_RIGHT)
        *out_state = state < nb_states - 1 ? state+1 : state;
        if(state == nb_states - 2) {
            rew = rewards[REW_WIN];
        } else {
            rew = rewards[REW_NEUTRAL];
        }
    }
    if(state == 0 || state == nb_states - 1) {
        rew = 0;
    }

    return rew;
}

static void test_iterative_fun()
{
    {
        double esperance[nb_states] = { 0 };
        double theta = 0.0001;
        double gamma = 0.9999;

        double policy_all_right[nb_states][nb_actions];
        for(int s = 1; s < nb_states ; s++) {
            policy_all_right[s][ACTION_LEFT] = 0;
            policy_all_right[s][ACTION_RIGHT] = 1;
        }

        //test stratégie all right sur lineworld avec policy iterative
        ial_eval_policy_iterative_fun(nb_states, nb_actions, policy_all_right, reward, theta, gamma, esperance);
        print_esperance(esperance,nb_states);
    }

}

void print_esperance(double *esperance,int nbstate){
    for (int s = 0 ; s<nbstate;s++){
        printf("%lf\t",esperance[s]);
    }
    printf("\n");
}

int main(void) 
{
    puts("iterative init");
    long s = get_nanoseconds();
    test_iterative();
    printf("spend %ld\n", get_nanoseconds() - s);
    puts("iterative fun");
    s = get_nanoseconds();
    test_iterative_fun();
    printf("spend %ld\n", get_nanoseconds() - s);

}
