import tensorflow as tf
from tensorforce.environments import Environment
from tensorforce.agents import Agent
import numpy as np

# Verificar que TensorFlow funciona
print("TensorFlow versión:", tf.__version__)

# Definir un entorno simple para prueba
class SimpleEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.state = np.array([0.0], dtype=np.float32)

    def states(self):
        return dict(type='float', shape=(1,))

    def actions(self):
        return dict(type='float', shape=(1,))

    def reset(self):
        self.state = np.array([0.0], dtype=np.float32)
        return self.state

    def execute(self, actions):
        self.state += actions
        reward = -abs(self.state[0])
        terminal = False
        return self.state, reward, terminal

# Inicializar entorno y agente
environment = SimpleEnvironment()

agent = Agent.create(
    agent='ppo',  # Puede ser DQN, PPO, etc.
    environment=environment,
    batch_size=10,
    learning_rate=0.001
)

# Ejecutar una prueba con el agente
for _ in range(100):
    state = environment.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done = environment.execute(action)
        agent.observe(reward=reward, terminal=done)

print("Ejecución completada en KV260")