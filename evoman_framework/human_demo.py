################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from evoman.environment import Environment

experiment_name = 'test'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with human player and static enemies
for en in range(1, 9):
    env = Environment(experiment_name=experiment_name,
                      enemymode='static',
                      speed="normal",
                      sound="off",
                      fullscreen=True,
                      use_joystick=True,
                      playermode='human',
                      visuals=True)
    env.update_parameter('enemies', [en])
    env.play()

