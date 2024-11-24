#include <stdlib.h>
#include <stdio.h>


#include "ialib.h"

#include "line_world.h"

double theta = 0.0001;
double gamma = 0.9999;

static void test_iterative()
{

    enum{ nb_states = 10 };
    struct ial_env line_env = { 0 };
    generate_line_world(&line_env, nb_states);

    double state_value_for_policy[nb_states] = {0};

    {
        double policy_all_left[nb_states][nb_actions];
        for(int s = 0; s < nb_states ; s++) {
            policy_all_left[s][ACTION_LEFT] = 1;
            policy_all_left[s][ACTION_RIGHT] = 0;
        }

        ///test stratégie all left sur lineworld avec policy iterative
        ial_eval_policy_iterative(
                nb_states, nb_actions, policy_all_left, nb_rewards, rewards,
                state_value_for_policy, theta, gamma,
                                               &line_env);
        print_esperance(state_value_for_policy, nb_states);
    }

    {
        double policy_all_right[nb_states][nb_actions];
        for(int s = 0; s < nb_states ; s++) {
            policy_all_right[s][ACTION_LEFT] = 0;
            policy_all_right[s][ACTION_RIGHT] = 1;
        }

        //test stratégie all right sur lineworld avec policy iterative
        ial_eval_policy_iterative(nb_states, nb_actions, policy_all_right, nb_rewards,
                                  rewards, state_value_for_policy, theta, gamma, &line_env);
        print_esperance(state_value_for_policy, nb_states);
    }

    destroy_line_world(&line_env);
}

const double reward(const int nb_states, const int state, const int action, int* out_state)
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
    /** plus correct mais double le temps d'execution
    if(state == 0 || state == nb_states - 1) {
        rew = 0;
    }
    */

    return rew;
}

static void test_iterative_fun()
{
    {
        enum{ nb_states = 10 };

        double esperance[nb_states] = { 0 };

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

void test_policy_iteration()
{
    enum{ nb_states = 10 };

    double pi[nb_states][nb_actions] = { 0 };
    double esperance[nb_states] = { 0 };

    ial_policy_iteration_naive(nb_states, nb_actions, reward, theta, gamma, pi, esperance);
    print_esperance(esperance,nb_states);
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
    /*
    puts("policy iter");
    s = get_nanoseconds();
    test_policy_iteration();
    printf("spend %ld\n", get_nanoseconds() - s);
     */
}
