import gym
from stable_baselines3 import PPO

# Crear el entorno de entrenamiento
env = gym.make("CartPole-v1")

# Inicializar el modelo PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entrenar el modelo
model.learn(total_timesteps=10000)

# Guardar el modelo
model.save("ppo_cartpole")

# Cargar el modelo entrenado
model = PPO.load("ppo_cartpole")

# Evaluar el modelo
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

env.close()
