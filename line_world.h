//
// Created by ivo on 24/11/2024.
//

#ifndef IALIB_LINE_WORLD_H
#define IALIB_LINE_WORLD_H

enum actions {ACTION_LEFT, ACTION_RIGHT};
enum rewards {REW_LOSS, REW_NEUTRAL, REW_WIN};

extern int actions[2];
enum {nb_actions = sizeof actions / sizeof actions[0]};
extern double rewards[3];
enum {nb_rewards = sizeof rewards / sizeof rewards[0]};


void generate_line_world(struct ial_env*, int);
void destroy_line_world(struct ial_env*);

#endif //IALIB_LINE_WORLD_H
