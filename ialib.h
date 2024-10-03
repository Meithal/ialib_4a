
char* ial_version(void);




void ial_ballman(
    int nb_states,
    int nb_actions,
    float (* pi)[nb_states][nb_actions],
    float (* env_S)[nb_states],
    float (* env_A)[nb_actions],
    int nb_rewards,
    int (* env_R)[nb_rewards],
    float (*env_p)[nb_states][nb_actions][nb_states][nb_rewards]
    )
{

}