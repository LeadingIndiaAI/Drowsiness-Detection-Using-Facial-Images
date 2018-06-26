import random
import os

path = 'music'
files = os.listdir(path)
play = random.choice(files)
print(os.path.abspath(play))
