
int get_random_int();
/**
 * Retourne un floattant entre 0 et 1
 */
double get_rand_float();
double max(double a, double b);
double min(double a, double b);


char* ial_version(void);

void ial_start(void);


void ial_eval_policy_iterative(
    int nb_states,
    int nb_actions,
    float pi[nb_states][nb_actions],
    float env_S[nb_states],
    float env_A[nb_actions],
    int nb_rewards,
    int env_R[nb_rewards],
    float env_p[nb_states][nb_actions][nb_states][nb_rewards],
    float esperance[nb_states],
    float theta,
    float gamma
    );