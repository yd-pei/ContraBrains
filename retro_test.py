import retro
import gym

env = retro.make(game = 'Contra-Nes')

env.reset()
env.render()
