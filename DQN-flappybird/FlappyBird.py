# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------
import time
t1 = time.time()
import csv
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
from BrainNEC import BrainNEC
import numpy as np
import os

train_frame = 2000000

# preprocess raw image to 80*80 gray image
def preprocess(observation):
        observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
        return np.reshape(observation,(80,80,1))

def playFlappyBird1():
        directory = 'saved_networks_dqn'
        if not os.path.exists(directory):
                os.makedirs(directory)
        # Step 1: init BrainDQN
        actions = 2
        brain = BrainDQN(actions)
        # Step 2: init Flappy Bird Game
        flappyBird = game.GameState()
        # Step 3: play game
        # Step 3.1: obtain init state
        action0 = np.array([1,0])  # do nothing
        observation0, reward0, terminal = flappyBird.frame_step(action0)
        observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
        brain.setInitState(observation0)

        # Step 3.2: run the game
        iii = 0
        f = open('lifetime_DQN.csv','w',newline='')
        ww = csv.writer(f)
        lf_1 = 0
        lf_2 = 0
        lf_time = 0
        #while 1!=0:
        while iii < train_frame:
                action = brain.getAction()
                nextObservation,reward,terminal = flappyBird.frame_step(action)
                #flappyBird1.frame_step([1,0])
                nextObservation1 = preprocess(nextObservation)
                brain.setPerception(nextObservation1,action,reward,terminal)
                iii += 1
                if terminal == True:
                        print ('----')
                        lf_2 = iii
                        lf_time = lf_2 - lf_1
                        ww.writerow(['no. of step',iii,'lifetime',lf_time])
                        lf_1 = lf_2
                #time.sleep(0.5)
        f.close()
        
def playFlappyBird2():
        directory = 'saved_networks_nec'
        if not os.path.exists(directory):
                os.makedirs(directory)
        # Step 1: init BrainDQN
        actions = 2
        brain = BrainNEC(actions)
        # Step 2: init Flappy Bird Game
        flappyBird = game.GameState()
        # Step 3: play game
        # Step 3.1: obtain init state
        action0 = np.array([1,0])  # do nothing
        observation0, reward0, terminal = flappyBird.frame_step(action0)
        observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
        brain.setInitState(observation0)

        # Step 3.2: run the game
        iii = 0
        f = open('lifetime_NEC.csv','w',newline='')
        ww = csv.writer(f)
        lf_1 = 0
        lf_2 = 0
        lf_time = 0
        #while 1!=0:
        while iii < train_frame:
                action = brain.getAction()
                nextObservation,reward,terminal = flappyBird.frame_step(action)
                #flappyBird1.frame_step([1,0])
                nextObservation1 = preprocess(nextObservation)
                brain.setPerception(nextObservation1,action,reward,terminal)
                iii += 1
                print (iii ,action , reward , terminal)
                if terminal == True:
                        print ('----')
                        lf_2 = iii
                        lf_time = lf_2 - lf_1
                        ww.writerow(['no. of step',iii,'lifetime',lf_time])
                        lf_1 = lf_2
                x = np.zeros((80,80,1),dtype=np.uint8)
                print(brain.getkey(x))
                #time.sleep(0.5)
        f.close()

def main():
        playFlappyBird2()
        playFlappyBird1()        

if __name__ == '__main__':
        print ( time.time() - t1)
        main()
        print (time.time() - t1)
