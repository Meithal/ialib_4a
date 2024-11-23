import random
from collections import defaultdict

A = [0, 1] # 0 : Gauche, 1 : Droite
S = [0, 1, 2, 3, 4] # l'agent est dans la case i
R = [-1.0, 0.0, 1.0] # recompenses immédiates
T = [0, 4] # etats finaux
p = [
        [
            [
                [0.0 for r_index in range(len(R))] for s_next in range(len(S))
            ] for a in range(len(A))
        ] for s in range(len(S))
]

for s in [1, 2]:
  p[s][1][s+1][1] = 1.0  ## on ne gagne aucun reward si on se deplace a droite depuis 1 ou 2 
for s in [2, 3]:
  p[s][0][s-1][1] = 1.0  ## on ne gagne aucun reward si on se deplace a gauche depuis 1 ou 2 
p[3][1][4][2] = 1.0  ## on gagne si on se deplace a droite et qu'on est a case 3
p[1][0][0][0] = 1.0  ## on perd si on se deplace a gauche et qu'on est a la case 1

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
  print(V)
  return V

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

  print(V)
  return V


def monte_carlo_es(env_S, env_A, env_R, env_p, gamma=0.9, num_episodes=10000):
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


def monte_carlo_on_policy(env_S, env_A, env_R, env_p, policy, gamma=0.9, num_episodes=10000):
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


def monte_carlo_off_policy_control(env_S, env_A, env_R, env_p, gamma=0.9, num_episodes=10000):
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



# Exécuter Monte Carlo Exploring Starts
policy, Q = monte_carlo_es(S, A, R, p)

# Afficher la politique optimisée
print("Politique optimisée :", policy)

# Afficher les valeurs de Q pour chaque état-action
print("\nValeurs Q pour chaque paire état-action:")
for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)

policy = {s: random.choice(A) for s in S}

policy, Q = monte_carlo_on_policy(S, A, R,p, policy)

# Afficher la politique optimisée
print("Politique optimisée :", policy)

# Afficher les valeurs Q
for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)

policy, Q = monte_carlo_off_policy_control(S, A, R,p)

# Afficher la politique optimisée
print("Politique optimisée :", policy)

# Afficher les valeurs Q
for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)





pi_right = [[1.0 if a == 1 else 0.0 for a in A] for s in S]
iterative_policy_evaluation(pi_right, S, A, R, p, T)
pi_left = [[0.0 if a == 1 else 1.0 for a in A] for s in S]
iterative_policy_evaluation(pi_left, S, A, R, p, T)

Value_Iteration(S,A,R,p)

Q = sarsa(S, A, R, p)

# Afficher les valeurs Q
for state, actions in Q.items():
    print(f"Valeurs Q pour l'état {state} :", actions)