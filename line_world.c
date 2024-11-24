//
// Created by ivo on 24/11/2024.
//

#include "stdlib.h"

#include "ialib.h"

#include "line_world.h"


int actions[] = {ACTION_LEFT, ACTION_RIGHT};
double rewards[] = {[REW_LOSS] = -1, [REW_NEUTRAL] = 0, [REW_WIN]=1};



void generate_line_world(struct ial_env* env, int nb_states) {
    env->_nb_actions = nb_actions;
    env->_nb_states = nb_states;
    env->_nb_rewards = nb_rewards;

    double (*transitions)[nb_states][nb_actions][nb_states][nb_rewards] = malloc(sizeof (double[nb_states][nb_actions][nb_states][nb_rewards]));

    for (int s = 1; s <= nb_states-3; s++) {
        (*transitions)[s][ACTION_RIGHT][s+1][REW_NEUTRAL] = 1.0;
    }

    for (int s = 2; s <= nb_states-2; s++) {
        (*transitions)[s][ACTION_LEFT][s-1][REW_NEUTRAL] = 1.0;
    }

    (*transitions)[nb_states-2][ACTION_RIGHT][nb_states-1][REW_WIN] = 1.0;
    (*transitions)[1][ACTION_LEFT][0][REW_LOSS] = 1.0;

    env->_probabs_transitions = transitions;
}

void destroy_line_world(struct ial_env* env) {
    free(env->_probabs_transitions);
}