#include <stdlib.h>
#include <stdio.h>


#include "ialib.h"

enum actions {ACTION_LEFT, ACTION_RIGHT, ACTION_BOTTOM,ACTION_UP};
enum rewards {REW_LOSS, REW_NEUTRAL, REW_WIN};

enum{nb_row = 5, nb_col = 5};
enum {nb_states = nb_col*nb_row};
int actions[] = {ACTION_LEFT, ACTION_RIGHT, ACTION_BOTTOM, ACTION_UP};
enum {nb_actions = sizeof actions / sizeof actions[0]};
double rewards[] = {[REW_LOSS] = -1, [REW_NEUTRAL] = 0, [REW_WIN]=1};
enum {nb_rewards = sizeof rewards / sizeof rewards[0]};

double theta = 0.0001;
double gamma = 0.9999;

static void test_iterative()
{
    double (*transitions)[nb_states][nb_actions][nb_states][nb_rewards] = malloc(sizeof (double[nb_states][nb_actions][nb_states][nb_rewards]));

    for(int x = 0 ; x<nb_col ;x++) {
        for(int y = 0 ; y<nb_row;y++) {
            if((x!=nb_col-1 || y!=0) && (x!=nb_col-1 || y!=nb_row-1)){//on evite les etats finaux (en haut à droite et en bas à droite)
                if(x!=0) {
                    (*transitions)[s][ACTION_LEFT][s-1][REW_NEUTRAL] = 1.0;
                }
                if(x!=nb_col-1) {
                    (*transitions)[s][ACTION_RIGHT][s+1][REW_NEUTRAL] = 1.0;
                    
                }
                if(y!=0) {
                    (*transitions)[s][ACTION_UP][s-nb_col][REW_NEUTRAL] = 1.0;
                }
                if(y!=nb_row-1) {
                    (*transitions)[s][ACTION_BOTTOM][s+nb_col][REW_NEUTRAL] = 1.0;
                }
            }
        }
    }

    (*transitions)[nb_col-2][ACTION_RIGHT][nb_col-1][REW_WIN] = 1.0;
    (*transitions)[nb_col-1 +nb_col][ACTION_UP][nb_col-1][REW_WIN] = 1.0;
    (*transitions)[nb_states-2][ACTION_RIGHT][nb_states-1][REW_WIN] = 1.0;
    (*transitions)[nb_states-1 -nb_col][ACTION_BOTTOM][nb_states-1][REW_WIN] = 1.0;

    double esperance[nb_states] = {0};

    double theta = 0.0001;
    double gamma = 0.9999;

    {
        double policy_all_left[nb_states][nb_actions]; //policy en passant par le coin en bas à gauche
        for(int x = 0; x < nb_col ; x++) {
            for(int y=0;y<nb_row;y++){
                if(y<nb_row-1){
                    policy_all_left[s][ACTION_LEFT] = 0;
                    policy_all_left[s][ACTION_RIGHT] = 0;
                    policy_all_left[s][ACTION_BOTTOM] = 1;
                    policy_all_left[s][ACTION_UP] = 0;
                }else{
                    policy_all_left[s][ACTION_LEFT] = 0;
                    policy_all_left[s][ACTION_RIGHT] = 1;
                    policy_all_left[s][ACTION_BOTTOM] = 0;
                    policy_all_left[s][ACTION_UP] = 0;
                }
            }
        }
        

        ///test stratégie all left sur lineworld avec policy iterative
        ial_eval_policy_iterative(
                nb_states, nb_actions, policy_all_left, nb_rewards, rewards,
                transitions, esperance, theta, gamma);
        print_esperance(esperance,nb_states);
    }

    free(transitions);
}

double reward(int state, int action, int* out_state)
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
}
