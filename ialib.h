
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
    double pi[nb_states][nb_actions],
    int env_S[nb_states],
    int env_A[nb_actions],
    int nb_rewards,
    double env_R[nb_rewards],
    double (*env_p)[nb_states][nb_actions][nb_states][nb_rewards],
    double esperance[nb_states],
    double theta,
    double gamma
    );

void ial_policy_iteration(
    int nb_states,
    int nb_actions,
    int env_S[nb_states],
    int env_A[nb_actions],
    int nb_rewards,
    double env_R[nb_rewards],
    double (*env_probas)[nb_states][nb_actions][nb_states][nb_rewards],
    double esperance[nb_states],
    double theta,
    double gamma);