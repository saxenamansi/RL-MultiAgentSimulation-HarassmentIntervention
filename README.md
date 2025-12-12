# RL-MultiAgentSimulation-HarassmentIntervention
Online Sexual Harassment Multiagent Simulation with Reinforcement Learning-Based Interventions

# Abstract

Online sexual harassment is a persistent problem on digital platforms, and timely intervention is critical to prevent harm to victims. 
We study the problem of automated online harassment intervention by modeling it as a sequential decision-making task. 
We simulate a multiagent environment in which a harasser and a victim interact through text messaging, and an intervener agent decides when and how to intervene using platform-style notices. 
The environment is driven by large language models (LLMs) for dialogue generation and an external moderation model for estimating harassment severity. 
We formulate the problem as a Markov Decision Process and train a reinforcement learning (RL) agent to minimize harassment severity over time while avoiding unnecessary or delayed interventions. 
This work demonstrates the feasibility of RL for studying calibrated intervention policies in complex, language-driven social environments.
