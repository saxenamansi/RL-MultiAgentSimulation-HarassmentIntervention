import numpy as np

def persona_sampler(role, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if role == "harasser":
        O = np.random.rand()
        C = np.random.beta(2, 7)
        E = np.random.rand()
        A = np.random.beta(2, 7)
        N = np.random.beta(7, 2)
        M = np.random.beta(7, 2)
        P = np.random.beta(7, 2)
        R = np.random.beta(7, 2)
        return {"O": O, "C": C, "E": E, "A": A, "N": N, "M": M, "P": P, "R": R}

    elif role == "victim":
        traits = np.random.rand(8)
        return {
            "O": traits[0],
            "C": traits[1],
            "E": traits[2],
            "A": traits[3],
            "N": traits[4],
            "M": traits[5],
            "P": traits[6],
            "R": traits[7],
        }

    elif role == "intervener":
        return None

    else:
        raise ValueError("Role must be one of: harasser, victim, intervener")


def get_persona():
    h_traits = persona_sampler("harasser", seed=None)
    v_traits = persona_sampler("victim", seed=None)
    return h_traits, v_traits
