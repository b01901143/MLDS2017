import os
import cv2
import random
import numpy as np
import tensorflow as tf
from collections import deque
from agent import *
from environment import *

MAX_TO_KEEP = 150
STORE_PER_STEP = 25000
MODEL_DIR = "./models/"
SUMMARY_DIR = "./summary/"
DISPLAY_PER_STEP = 1000
LEARNING_RATE = 1e-6
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
OBSERVE_STEP = 10000
EXPLORE_STEP = 3000000
REPLAY_MEMORY_LENGTH = 50000
START_EPSILON = 0.1
FINAL_EPSILON = 0.0001
GAMMA = 0.99
BATCH_SIZE = 32

def train():
    #agent
    agent = Agent()
    s_tensor, q_tensor = agent.build_agent()
    a_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=(None, NUM_ACTIONS),
        name="a_tensor"
    )
    y_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=(None),
        name="y_tensor"
    )
    q_out_tensor = tf.reduce_sum(
        input_tensor=tf.multiply(
            q_tensor,
            a_tensor
        ),
        axis=-1,
        keep_dims=False,
        name="q_out_tensor"
    )
    loss = tf.reduce_mean(
        input_tensor=tf.square(
            y_tensor - q_out_tensor
        ),
        axis=-1,
        keep_dims=False,
        name="loss"
    )
    tf.summary.histogram("s_tensor", s_tensor)
    tf.summary.histogram("q_tensor", q_tensor)
    tf.summary.histogram("a_tensor", a_tensor)
    tf.summary.histogram("y_tensor", y_tensor)
    tf.summary.histogram("q_out_tensor", q_out_tensor)
    tf.summary.scalar("loss", loss)    
    summary_tensor = tf.summary.merge_all()      
    global_step = tf.contrib.framework.get_or_create_global_step()
    trainable_var_list = tf.get_collection(
        key=tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=None
    )
    train_op = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE,
        beta1=BETA_1,
        beta2=BETA_2,
        epsilon=EPSILON,
        use_locking=False,
        name="train_op"
    ).minimize(
        loss=loss,
        global_step=global_step,
        var_list=trainable_var_list
    )
    init_op = tf.global_variables_initializer()
    #session
    session = tf.InteractiveSession()
    session.run(init_op)
    #summarier
    summarier = tf.summary.FileWriter(SUMMARY_DIR)
    #saver
    saver = tf.train.Saver()
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    #pygame
    pygame.init()
    pygame_display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption("flappy bird")
    pygame_time_clock = pygame.time.Clock()
    #environment
    score, score_pos_x_offset = 0, 0
    images_dict, sounds_dict, masks_dict = load_assets()
    environment = Environment(images_dict, sounds_dict, masks_dict)
    #train
    replay_memory = deque()
    x_t_experience = pygame.surfarray.array3d(pygame.display.get_surface())          
    x_t_experience = cv2.cvtColor(cv2.resize(x_t_experience, (SCREEN_HEIGHT, SCREEN_WIDTH)), cv2.COLOR_BGR2GRAY)
    _, x_t_experience = cv2.threshold(x_t_experience, 1, 255, cv2.THRESH_BINARY)
    s_t_experience = np.stack((x_t_experience, x_t_experience, x_t_experience, x_t_experience), axis=2)
    epsilon, step = START_EPSILON, 0
    while True:
        #display
        pygame_display.blit(images_dict["background"], (environment.background_pos_x, environment.background_pos_y))
        for pos_x, pos_y in zip(environment.pipe_pos_x_list, environment.pipe_pos_y_list):
            pygame_display.blit(images_dict["pipe"][0], (pos_x["up"], pos_y["up"]))
            pygame_display.blit(images_dict["pipe"][1], (pos_x["down"], pos_y["down"])) 
        pygame_display.blit(images_dict["base"], (environment.base_pos_x, environment.base_pos_y))
        pygame_display.blit(images_dict["player"][environment.player_index], (environment.player_pos_x, environment.player_pos_y))
        score_pos_x_offset_temp = 0
        for digit in list(str(score)):
            score_pos_x_offset_temp += images_dict["number"][int(digit)].get_width()
        score_pos_x_offset = (environment.background_width - score_pos_x_offset_temp) / 2
        for digit in list(str(score)):
            pygame_display.blit(images_dict["number"][int(digit)], (score_pos_x_offset, environment.background_height * 0.1))
            score_pos_x_offset += images_dict["number"][int(digit)].get_width()     
        pygame.display.update()
        pygame_time_clock.tick(FRAME_PER_SEC)
        #update replay memory
        q_t_experience = session.run(
            q_tensor,
            feed_dict={
                s_tensor: [ s_t_experience ]
            }
        )[0]
        a_t_experience = np.zeros(NUM_ACTIONS, )
        if random.random() < epsilon:
            a_t_experience[random.randrange(NUM_ACTIONS)] = 1
        else:
            a_t_experience[np.argmax(q_t_experience)] = 1        
        player_object = {
            "pos_x": environment.player_pos_x,
            "pos_y": environment.player_pos_y,
            "width": environment.player_width,
            "height": environment.player_height,
            "index": environment.player_index
        }
        pipe_objects = {
            "pos_x": environment.pipe_pos_x_list,
            "pos_y": environment.pipe_pos_y_list,
            "width": environment.pipe_width,
            "height": environment.pipe_height
        }
        base_object = {
            "pos_x": environment.base_pos_x,
            "pos_y": environment.base_pos_y,
            "width": environment.base_width,
            "height": environment.base_height
        }
        r_t_experience = 0.1 
        t_t_experience = False
        if check_score(player_object, pipe_objects) == True:
            sounds_dict["point"].play()
            score += 1
            r_t_experience = 1  
        if check_object_crash(player_object, pipe_objects, base_object, masks_dict) == True:
            sounds_dict["hit"].play()
            sounds_dict["die"].play()
            score = 0
            r_t_experience = -1
            t_t_experience = True
            environment.__init__(images_dict, sounds_dict, masks_dict)
        else:
            environment.next_frame_step(a_t_experience)        
        x_t_1_experience = pygame.surfarray.array3d(pygame.display.get_surface())
        x_t_1_experience = cv2.cvtColor(cv2.resize(x_t_1_experience, (SCREEN_HEIGHT, SCREEN_WIDTH)), cv2.COLOR_BGR2GRAY)
        _, x_t_1_experience = cv2.threshold(x_t_1_experience, 1, 255, cv2.THRESH_BINARY)
        x_t_1_experience = np.reshape(x_t_1_experience, (SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        s_t_1_experience = np.append(x_t_1_experience, s_t_experience[:, :, :SCREEN_LENGTH-1], axis=2)
        replay_memory.append((s_t_experience, a_t_experience, r_t_experience, s_t_1_experience, t_t_experience))
        if len(replay_memory) > REPLAY_MEMORY_LENGTH:
            replay_memory.popleft()
        #update epsilon
        if step > OBSERVE_STEP and epsilon > FINAL_EPSILON:
            epsilon -= (START_EPSILON - FINAL_EPSILON) / EXPLORE_STEP
        #update train_op
        if step > OBSERVE_STEP:
            data_batch = random.sample(replay_memory, BATCH_SIZE)
            s_t_batch, a_t_batch, r_t_batch, s_t_1_batch, t_t_batch = zip(*data_batch)
            q_t_1_batch = session.run(
                q_tensor, 
                feed_dict={
                    s_tensor: s_t_1_batch
                }
            )
            y_t_batch = [ r_t if t_t else r_t + GAMMA * np.max(q_t_1) for r_t, t_t, q_t_1 in zip(r_t_batch, t_t_batch, q_t_1_batch) ]
            summary_value, _ = session.run(
                [ summary_tensor, train_op ],
                feed_dict={
                    s_tensor: s_t_batch,
                    a_tensor: a_t_batch,
                    y_tensor: y_t_batch
                }
            )
            summarier.add_summary(summary_value, step)
            if step % DISPLAY_PER_STEP == 0:
                print "Step:", step, "Epsilon:", epsilon, "Q_max:", np.max(q_t_experience)
        s_t_experience = s_t_1_experience
        step += 1
        #store
        if step % STORE_PER_STEP == 0:
            saver.save(session, MODEL_DIR, global_step=step)

if __name__ == "__main__":
    train()
