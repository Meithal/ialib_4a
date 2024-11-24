import random
import numpy as np
from collections import defaultdict
import time

A = [0, 1, 2, 3] # 0 : Gauche, 1 : Droite, 2 : Bas , 3 : Haut
nb_actions = 4
nb_col = 5 #CHANGEABLE
nb_row = 5 #CHANGEABLE
nb_states = nb_col*nb_row
S = list(range(nb_col*nb_row))
R = [-3.0, 0.0, 1.0] # recompenses immédiates
nb_rewards = 3
T = [0, nb_col*nb_row-1] # etats finaux
p = np.zeros((nb_states, nb_actions, nb_states, nb_rewards))

for x in range(nb_col):
    for y in range(nb_row):
        s = y * nb_col + x  # Calcul de l'état actuel (index linéaire)
        if (x != nb_col - 1 or y != 0) and (x != nb_col - 1 or y != nb_row - 1):  # Éviter les états finaux
            if x != 0:
                p[s, 0, s - 1, 1] = 1.0
            if x != nb_col - 1:
                p[s, 1, (s + 1), 1] = 1.0
            if y != 0:
                p[s, 3, s - nb_col, 1] = 1.0
            if y != nb_row - 1:
                p[s, 2, s + nb_col, 1] = 1.0

# Définition des transitions pour les états finaux
p[nb_col - 2, 1, nb_col - 1, 0] = 1.0
p[nb_col - 1 + nb_col, 3, nb_col - 1, 0] = 1.0
p[nb_states - 2, 1, nb_states - 1, 2] = 1.0
p[nb_states - 1 - nb_col, 2, nb_states - 1, 2] = 1.0


def iterative_policy_evaluation(
    pi,
    env_S,
    env_A,
    env_R,
    env_p,
    env_T = None,
    theta=0.00001,
    gamma=0.9999,
    V = None
    ):
  if V is None:
    if env_T is None:
      raise "env T is None"
    V = [0.0 if s in env_T else random.random() for s in env_S]

  while True:
    delta = 0.0
    for s in env_S:
      v_prev = V[s]
      total = 0.0
      for a in env_A:
        for s_p in env_S:
          for r_index in range(len(env_R)):
            r = env_R[r_index]
            total += pi[s][a] * env_p[s][a][s_p][r_index] * (r + gamma * V[s_p])
      V[s] = total
      delta = max(delta, abs(v_prev - V[s]))
    if delta < theta:
      break
  return V


def policy_iteration(env_S, env_A, env_R, env_p, env_T=None, gamma=0.9999, theta=0.00001):
    """
    Implémente l'algorithme de Policy Iteration.

    :param env_S: Liste des états.
    :param env_A: Liste des actions.
    :param env_R: Liste des récompenses.
    :param env_p: Proba de transition (env_p[s][a][s'][r]).
    :param env_T: États terminaux (facultatif).
    :param gamma: Facteur de discount.
    :param theta: Critère de convergence.
    :return: La politique optimale et la fonction de valeur.
    """
    if env_T is None:
        raise ValueError("env_T doit être fourni pour définir les états terminaux.")
    
    # Initialisation
    V = [0.0 if s in env_T else random.random() for s in env_S]
    pi = {s: random.choice(env_A) for s in env_S if s not in env_T}
    
    while True:
        # --- Policy Evaluation ---
        while True:
            delta = 0.0
            for s in env_S:
                if s in env_T:
                    continue
                v_prev = V[s]
                a = pi[s]
                total = 0.0
                for s_p in env_S:
                    for r_index in range(len(env_R)):
                        r = env_R[r_index]
                        total += env_p[s][a][s_p][r_index] * (r + gamma * V[s_p])
                V[s] = total
                delta = max(delta, abs(v_prev - V[s]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        policy_stable = True
        for s in env_S:
            if s in env_T:
                continue
            old_action = pi[s]

            # Trouver la meilleure action
            best_action = max(
                env_A,
                key=lambda a: sum(
                    env_p[s][a][s_p][r_index] * (env_R[r_index] + gamma * V[s_p])
                    for s_p in env_S for r_index in range(len(env_R))
                )
            )
            
            pi[s] = best_action
            if old_action != pi[s]:
                policy_stable = False

        if policy_stable:
            break

    return pi, V


def Value_Iteration(env_S,
  env_A,
  env_R,
  env_p,
  theta=0.00001,
  gamma=0.9999
  ):
  V = [0.0 for s in env_S]
  
  while True:
    delta = 0
    for s in env_S:
      v_prev = V[s]

      V[s] = max(
                sum(env_p[s][a][s_next][r_index] * (env_R[r_index] + gamma * V[s_next]) 
                    for s_next in env_S for r_index in range(len(R))) for a in env_A
            )
      delta =max(delta, abs(v_prev - V[s]))

    if delta < theta:
      break

  return V


def monte_carlo_es(env_S, env_A, env_R, env_p, gamma=0.999, num_episodes=10000):
    #print("Début de la fonction monte_carlo_es")
    # Initialisation des dictionnaires Q et Returns
    Q = defaultdict(lambda: [0.0 for _ in env_A])
    Returns = defaultdict(lambda: [[] for _ in env_A])  # Stocke des listes de retours pour chaque paire état-action
    policy = {s: random.choice(env_A) for s in env_S}   # Politique initiale
    #print("Initialisation de Q, Returns et policy terminée")

    def take_action(state, action, env_p, env_R):
      #print(f"Prendre une action : état {state}, action {action}")
      state_probabilities = [sum(env_p[state][action][s_next]) for s_next in range(len(env_p[state][action]))]
      #print(f"Probabilités d'état: {state_probabilities}")

      # Si aucune transition n'est possible depuis cet état avec cette action, on le considère comme terminal
      if sum(state_probabilities) == 0:
          #print(f"Aucune transition possible depuis l'état {state} avec action {action}. Considéré comme terminal.")
          return state, 0, True  # Fin de l'épisode forcé

      next_state = random.choices(range(len(state_probabilities)), weights=state_probabilities)[0]
      reward_index = max(range(len(env_R)), key=lambda r: env_p[state][action][next_state][r])
      reward = env_R[reward_index]
      #print(f"Transition vers état {next_state} avec récompense {reward}")
      return next_state, reward, next_state in T

    max_steps = 100  # Limite de pas par épisode

    for episode_num in range(num_episodes):
        state = random.choice(env_S)
        action = random.choice(env_A)
        episode = []
        
        done = False
        steps = 0  # Compteur de pas
        while not done and steps < max_steps:
            next_state, reward, done = take_action(state, action, env_p, env_R)
            episode.append((state, action, reward))
            state = next_state
            action = policy[state] if not done else None
            steps += 1  # Incrémente le compteur de pas
        #print("Épisode généré:", episode)

        # Calcul des retours et mise à jour de Q et de la politique
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            #print(f"Retour G à temps {t}: {G}")

            # Vérifie si la paire état-action est la première occurrence dans l'épisode
            if not any((state == x[0] and action == x[1]) for x in episode[:t]):
                Returns[state][action].append(G)
                Q[state][action] = sum(Returns[state][action]) / len(Returns[state][action])
                #print(f"Valeur Q[{state}][{action}] mise à jour: {Q[state][action]}")

                # Met à jour la politique pour être optimale par rapport à Q
                best_action = max(range(len(env_A)), key=lambda a: Q[state][a])
                policy[state] = best_action
                #print(f"Politique mise à jour pour l'état {state}: action {best_action}")

    #print("Politique finale:", policy)
    return policy, Q


def monte_carlo_on_policy(env_S, env_A, env_R, env_p, policy, gamma=0.999, num_episodes=10000):
    """
    Monte Carlo On-Policy First Visit pour un environnement donné.
    """
    # Initialisation de Q et Returns
    Q = defaultdict(lambda: [0.0 for _ in env_A])  # Valeurs Q
    Returns = defaultdict(lambda: [[] for _ in env_A])  # Stocke les retours pour chaque paire état-action

    def take_action(state, policy, env_p, env_R):
        """
        Effectue une action en suivant la politique donnée.
        """
        action = policy[state]  # Action choisie par la politique
        state_probabilities = [sum(env_p[state][action][s_next]) for s_next in range(len(env_S))]

        # Si aucune transition n'est possible, considère comme état terminal
        if sum(state_probabilities) == 0:
            return state, 0, True

        # Choisir le prochain état en fonction des probabilités
        next_state = random.choices(range(len(state_probabilities)), weights=state_probabilities)[0]

        # Trouver la récompense associée
        reward_index = max(
            range(len(env_R)),
            key=lambda r_idx: env_p[state][action][next_state][r_idx],
        )
        reward = env_R[reward_index]

        # Vérifier si le prochain état est terminal
        done = next_state in T
        return next_state, reward, done

    max_steps = 100  # Limite de pas par épisode

    for episode_num in range(num_episodes):
        # Choix d'un état initial aléatoire
        state = random.choice(env_S)
        episode = []

        # Génération d'un épisode en suivant la politique donnée
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = policy[state]  # Suit la politique existante
            next_state, reward, done = take_action(state, policy, env_p, env_R)
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        # Calcul des retours et mise à jour de Q
        G = 0
        visited = set()  # Pour traquer les premières visites
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # Vérifie si c'est la première visite de la paire état-action
            if (state, action) not in visited:
                visited.add((state, action))
                Returns[state][action].append(G)
                Q[state][action] = sum(Returns[state][action]) / len(Returns[state][action])

                # Mise à jour de la politique (greedy)
                best_action = max(range(len(env_A)), key=lambda a: Q[state][a])
                policy[state] = best_action

    return policy, Q


def monte_carlo_off_policy_control(env_S, env_A, env_R, env_p, gamma=0.999, num_episodes=10000):
    """
    Monte Carlo Off-Policy Control pour un environnement défini par S, A, R, et p.
    """
    Q = defaultdict(lambda: [0.0 for _ in env_A])  # Valeurs Q
    C = defaultdict(lambda: [0.0 for _ in env_A])  # Somme pondérée
    target_policy = {s: random.choice(env_A) for s in env_S}  # Politique cible initiale

    def take_action(state, behavior_policy):
        """
        Prend une action selon la politique comportementale (aléatoire ici).
        """
        action = behavior_policy(state)  # Choix d'une action
        next_states_probs = env_p[state][action]  # Probabilités de transition
        state_weights = [
            sum(next_states_probs[s_next]) for s_next in range(len(env_S))
        ]  # Somme des probabilités pour chaque état suivant

        if sum(state_weights) == 0:  # Si aucune transition possible, état terminal
            return state, 0, True

        # Choisir un état suivant en fonction des probabilités
        next_state = random.choices(range(len(state_weights)), weights=state_weights)[0]

        # Identifier la récompense associée
        reward_index = max(
            range(len(env_R)),
            key=lambda r_idx: env_p[state][action][next_state][r_idx],
        )  # Trouver la récompense dominante
        reward = env_R[reward_index]

        return next_state, reward, next_state in T

    def behavior_policy(state):
        """
        Politique comportementale qui choisit des actions uniformément au hasard.
        """
        return random.choice(env_A)

    max_steps = 100  # Nombre maximum de pas dans un épisode

    for episode_num in range(num_episodes):
        state = random.choice(env_S)  # État initial
        episode = []
        done = False
        steps = 0

        # Générer un épisode avec la politique comportementale
        while not done and steps < max_steps:
            action = behavior_policy(state)
            next_state, reward, done = take_action(state, behavior_policy)
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        # Calcul des retours pondérés hors politique
        G = 0
        W = 1.0  # Poids d'importance initial
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward  # Retour G accumulé

            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            # Mise à jour de la politique cible pour être greedy par rapport à Q
            best_action = max(range(len(env_A)), key=lambda a: Q[state][a])
            target_policy[state] = env_A[best_action]

            # Si la politique comportementale diverge de la politique cible, arrêtez
            if target_policy[state] != action:
                break

            # Met à jour W pour la prochaine itération
            prob_behavior = 1.0 / len(env_A)  # Politique uniforme
            W *= 1.0 / prob_behavior

    return target_policy, Q

def sarsa(env_S, env_A, env_R, env_p, gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=10000):
    """
    SARSA: On-policy Temporal Difference Learning
    """
    # Initialisation des valeurs Q
    Q = defaultdict(lambda: [0.0 for _ in env_A])

    def choose_action(state):
        """
        Choisit une action en suivant une politique ε-greedy.
        """
        if random.random() < epsilon:  # Exploration
            return random.choice(env_A)
        else:  # Exploitation
            return max(range(len(env_A)), key=lambda a: Q[state][a])

    def take_action(state, action, env_p, env_R):
        """
        Effectue une transition dans l'environnement.
        """
        state_probabilities = [sum(env_p[state][action][s_next]) for s_next in range(len(env_S))]
        
        # Si aucune transition n'est possible, considère comme terminal
        if sum(state_probabilities) == 0:
            return state, 0, True
        
        # Choisir le prochain état
        next_state = random.choices(range(len(state_probabilities)), weights=state_probabilities)[0]
        
        # Trouver la récompense correspondante
        reward_index = max(
            range(len(env_R)),
            key=lambda r: env_p[state][action][next_state][r],
        )
        reward = env_R[reward_index]
        done = next_state in T
        return next_state, reward, done

    max_steps = 100  # Limite de pas par épisode

    for episode in range(num_episodes):
        # Choix aléatoire de l'état initial
        state = random.choice(env_S)
        action = choose_action(state)  # Choisir la première action

        done = False
        steps = 0
        while not done and steps < max_steps:
            # Effectuer une transition
            next_state, reward, done = take_action(state, action, env_p, env_R)

            # Choisir la prochaine action selon la politique ε-greedy
            next_action = choose_action(next_state)

            # Mettre à jour Q(s, a) avec l'équation SARSA
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            # Préparer pour le prochain tour
            state = next_state
            action = next_action
            steps += 1

    return Q



print("Monte Carlo ES")
start_time = time.time()
policy, Q = monte_carlo_es(S, A, R, p)
end_time = time.time()  # Heure de fin
print(f"Temps d'exécution Monte Carlo ES : {end_time - start_time:.4f} secondes\n")

print("Politique optimisée :")
policy_grid = np.array([policy[s] if s in policy else -1 for s in range(nb_row * nb_col)]).reshape(nb_row, nb_col)
for row in policy_grid:
    print(row)

print("\nValeurs Q pour chaque paire état-action:")
for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)

policy = {s: random.choice(A) for s in S}

print("Monter Carlo On Policy")
start_time = time.time()
policy, Q = monte_carlo_on_policy(S, A, R,p, policy)
end_time = time.time()  # Heure de fin
print(f"Temps d'exécution Monter Carlo On Policy: {end_time - start_time:.4f} secondes\n")

print("Politique optimisée :")
policy_grid = np.array([policy[s] if s in policy else -1 for s in range(nb_row * nb_col)]).reshape(nb_row, nb_col)
for row in policy_grid:
    print(row)

for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)

print("Monte Carlo Off Policy")
start_time = time.time()
policy, Q = monte_carlo_off_policy_control(S, A, R,p)
end_time = time.time()  # Heure de fin
print(f"Temps d'exécution Monte Carlo Off Policy : {end_time - start_time:.4f} secondes\n")

print("Politique optimisée :")
policy_grid = np.array([policy[s] if s in policy else -1 for s in range(nb_row * nb_col)]).reshape(nb_row, nb_col)
for row in policy_grid:
    print(row)

for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)


policy_all_left = np.zeros((nb_states, nb_actions))
for x in range(nb_col):
    for y in range(nb_row):
        s = y * nb_col + x  # Calcul de l'état actuel (index linéaire)
        if y < nb_row-1:  # Si ce n'est pas la dernière colonne
            policy_all_left[s, 0] = 0  # ACTION_LEFT
            policy_all_left[s, 1] = 0  # ACTION_RIGHT
            policy_all_left[s, 2] = 1  # ACTION_BOTTOM
            policy_all_left[s, 3] = 0  # ACTION_UP
        else:  # Dernière rangée
            policy_all_left[s, 0] = 0  # ACTION_LEFT
            policy_all_left[s, 1] = 1  # ACTION_RIGHT
            policy_all_left[s, 2] = 0  # ACTION_BOTTOM
            policy_all_left[s, 3] = 0  # ACTION_UP

print("\nIterative policy evaluation (avec policy full down et right): \n")
start_time = time.time()
value_grid = np.array(iterative_policy_evaluation(policy_all_left, S, A, R, p, T)).reshape(nb_row, nb_col)
end_time = time.time()  # Heure de fin
print(f"Temps d'exécution iterative avec full down and right : {end_time - start_time:.4f} secondes\n")
for row in value_grid:
    print(row)

print("\nPolicy evaluation : ")
# Appel de la fonction
start_time = time.time()
optimal_policy, optimal_value = policy_iteration(S, A, R, p, T, 0.9999, 0.00001)
end_time = time.time()  # Heure de fin
print(f"Temps d'exécution Policy evaluation : {end_time - start_time:.4f} secondes\n")
# Affichage des résultats
print("Politique optimale :")
policy_grid = np.array([optimal_policy[s] if s in optimal_policy else -1 for s in range(nb_row * nb_col)]).reshape(nb_row, nb_col)
for row in policy_grid:
    print(row)
print("\nValeurs optimales :")
value_grid = np.array(optimal_value).reshape(nb_row, nb_col)
for row in value_grid:
    print(row)


print("\nValue Iteration : \n")
start_time = time.time()
value_grid = np.array(Value_Iteration(S,A,R,p)).reshape(nb_row, nb_col)
end_time = time.time()  # Heure de fin
print(f"Temps d'exécution Value Iteration : {end_time - start_time:.4f} secondes\n")
for row in value_grid:
    print(row)


print("\nSarsa")
start_time = time.time()
Q = sarsa(S, A, R, p)
end_time = time.time()  # Heure de fin
print(f"Temps d'exécution Sarsa : {end_time - start_time:.4f} secondes\n")
print(Q)
# Afficher les valeurs Q
for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)