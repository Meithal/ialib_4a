import random

A = [0, 1] # 0 : Gauche, 1 : Droite
S = [0, 1, 2, 3, 4] # l'agent est dans la case i
R = [-1.0, 0.0, 1.0] # recompenses immédiates
T = [0, 4] # etats finaux
p = [ # matrice de transition: le score obtenu pour chaque transition d'etat
    [
        [
            [
                0.0 for r_index in range(len(R))
            ] for s_p in range(len(S))
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
    gamma=0.99999,
    V=None
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

pi_right = [[1.0 if a == 1 else 0.0 for a in A] for s in S]
iterative_policy_evaluation(pi_right, S, A, R, p, T)
pi_left = [[0.0 if a == 1 else 1.0 for a in A] for s in S]
iterative_policy_evaluation(pi_left, S, A, R, p, T)