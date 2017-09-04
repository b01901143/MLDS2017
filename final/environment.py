import random
import pygame
import numpy as np
from itertools import cycle

IMAGE_BACKGROUND_PATH = "./assets/images/background-black.png"
IMAGE_BASE_PATH = "./assets/images/base.png"
IMAGE_PIPE_PATH = "./assets/images/pipe-green.png"
IMAGE_PLAYER_PATH = [
	"./assets/images/redbird-upflap.png",
	"./assets/images/redbird-midflap.png",
	"./assets/images/redbird-downflap.png"
]
IMAGE_NUM_PATH = [
	"./assets/images/0.png",
	"./assets/images/1.png",
	"./assets/images/2.png",
	"./assets/images/3.png",
	"./assets/images/4.png",
	"./assets/images/5.png",
	"./assets/images/6.png",
	"./assets/images/7.png",
	"./assets/images/8.png",
	"./assets/images/9.png"
]
SOUND_DIE_PATH = "./assets/sounds/die.ogg"
SOUND_HIT_PATH = "./assets/sounds/hit.ogg"
SOUND_POINT_PATH = "./assets/sounds/point.ogg"
DISPLAY_WIDTH = 288
DISPLAY_HEIGHT = 512
FRAME_PER_SEC = 30

def load_assets():
	images_dict, sounds_dict, masks_dict = {}, {}, {}
	images_dict["background"] = pygame.image.load(IMAGE_BACKGROUND_PATH).convert_alpha()
	images_dict["base"] = pygame.image.load(IMAGE_BASE_PATH).convert_alpha()
	images_dict["pipe"] = [ pygame.transform.rotate(pygame.image.load(IMAGE_PIPE_PATH).convert_alpha(), 180), pygame.image.load(IMAGE_PIPE_PATH).convert_alpha() ]
	images_dict["player"] = [ pygame.image.load(path).convert_alpha() for path in IMAGE_PLAYER_PATH ]
	images_dict["number"] = [ pygame.image.load(path).convert_alpha() for path in IMAGE_NUM_PATH ]
	sounds_dict["die"] = pygame.mixer.Sound(SOUND_DIE_PATH)
	sounds_dict["hit"] = pygame.mixer.Sound(SOUND_HIT_PATH)
	sounds_dict["point"] = pygame.mixer.Sound(SOUND_POINT_PATH)
	masks_dict["pipe"] = [ [ [ bool(image.get_at((i, j))[3]) for j in range(image.get_height()) ] for i in range(image.get_width()) ] for image in images_dict["pipe"] ]
	masks_dict["player"] = [ [ [ bool(image.get_at((i, j))[3]) for j in range(image.get_height()) ] for i in range(image.get_width()) ] for image in images_dict["player"] ]	
	return images_dict, sounds_dict, masks_dict

def check_score(player_object, pipe_objects):
	player_mid_pos_x = player_object["pos_x"] + player_object["width"] / 2
	for pos_x in pipe_objects["pos_x"]:
		pipe_mid_pos_x = pos_x["up"] + pipe_objects["width"] / 2
		if pipe_mid_pos_x <= player_mid_pos_x < pipe_mid_pos_x + 4:
			return True
	return False

def check_pixel_collision(player_rect, player_mask, pipe_rect, pipe_mask):
	overlap_rect = player_rect.clip(pipe_rect)
	if overlap_rect.width == 0 or overlap_rect.height == 0:
		return False
	x_1, y_1 = overlap_rect.x - player_rect.x, overlap_rect.y - player_rect.y
	x_2, y_2 = overlap_rect.x - pipe_rect.x, overlap_rect.y - pipe_rect.y
	for x in range(overlap_rect.width):
		for y in range(overlap_rect.height):
			if player_mask[x_1+x][y_1+y] and pipe_mask[x_2+x][y_2+y]:
				return True
	return False

def check_object_crash(player_object, pipe_objects, base_object, object_masks):
	if player_object["pos_y"] + player_object["height"] >= base_object["pos_y"] - 1:
		return True
	player_rect = pygame.Rect(
		player_object["pos_x"], player_object["pos_y"],
		player_object["width"], player_object["height"] 
	)
	player_mask = object_masks["player"][player_object["index"]]
	for pos_x, pos_y in zip(pipe_objects["pos_x"], pipe_objects["pos_y"]):
		pipe_up_rect = pygame.Rect(
			pos_x["up"], pos_y["up"],
			pipe_objects["width"], pipe_objects["height"]
		)
		pipe_down_rect = pygame.Rect(
			pos_x["down"], pos_y["down"],
			pipe_objects["width"], pipe_objects["height"]
		)
		pipe_up_mask = object_masks["pipe"][0]
		pipe_down_mask = object_masks["pipe"][1]
		up_collided = check_pixel_collision(player_rect, player_mask, pipe_up_rect, pipe_up_mask)
		down_collided = check_pixel_collision(player_rect, player_mask, pipe_down_rect, pipe_down_mask)	
		if up_collided or down_collided:
			return True
	return False

class Environment:
	def __init__(self, images_dict, sounds_dict, masks_dict):
		#shape
		self.background_width, self.background_height = images_dict["background"].get_width(), images_dict["background"].get_height()
		self.base_width, self.base_height = images_dict["base"].get_width(), images_dict["base"].get_height()
		self.pipe_width, self.pipe_height = images_dict["pipe"][0].get_width(), images_dict["pipe"][0].get_height()
		self.pipe_pos_y_offset_list = [20, 30, 40, 50, 60, 70, 80, 90]
		self.pipe_pos_y_gap = 100
		self.player_width, self.player_height = images_dict["player"][0].get_width(), images_dict["player"][0].get_height()		
		#pos
		self.background_pos_x, self.background_pos_y = 0, 0
		self.base_pos_x, self.base_pos_y = 0, self.background_height * 0.79
		pipe_pos_y_offset = [ self.pipe_pos_y_offset_list[random.randint(0, len(self.pipe_pos_y_offset_list) - 1)] + int(self.base_pos_y * 0.2) for _ in range(2) ]
		self.pipe_pos_x_list, self.pipe_pos_y_list = [ { "up": self.background_width * scale, "down": self.background_width * scale } for scale in np.arange(1.0, 2.0, 0.5) ], [ { "up": offset - self.pipe_height, "down": offset + self.pipe_pos_y_gap } for offset in pipe_pos_y_offset ]	
		self.player_pos_x, self.player_pos_y = int(self.background_width * 0.2), int((self.background_height - self.player_height) / 2)
		#vel
		self.base_vel_x = -4
		self.pipe_vel_x = -4
		self.player_vel_y = 0
		self.player_max_y = 0
		self.player_max_vel_y = 10
		self.player_min_vel_y = -8		
		#acc
		self.player_gra_acc_y = 1
		self.player_flap_acc_y = -10
		#others
		self.player_iter = 0
		self.player_index = 0
		self.player_index_list = cycle([0, 1, 2, 1])
	def next_frame_step(self, action):
		#others
		if (self.player_iter + 1) % 3 == 0:
			self.player_index = next(self.player_index_list)
		self.player_iter = (self.player_iter + 1) % 30
		#vel
		if action[1] == 1 and self.player_pos_y > -2 * self.player_height:
			self.player_vel_y = self.player_flap_acc_y        
		if self.player_vel_y < self.player_max_vel_y:
			self.player_vel_y += self.player_gra_acc_y 
		#pos
		self.base_pos_x = -((-(self.base_pos_x + self.base_vel_x)) % (self.base_width - self.background_width))
		for pair in self.pipe_pos_x_list:
			pair["up"] += self.pipe_vel_x
			pair["down"] += self.pipe_vel_x		
		self.player_pos_y += min(self.player_vel_y, self.base_pos_y - self.player_pos_y - self.player_height)
		if self.player_pos_y < 0:
			self.player_pos_y = 0
		#storage
		if 0 < self.pipe_pos_x_list[0]["up"] < 5:
			pipe_pos_y_offset = self.pipe_pos_y_offset_list[random.randint(0, len(self.pipe_pos_y_offset_list) - 1)] + int(self.base_pos_y * 0.2)
			self.pipe_pos_x_list.append({ "up": self.background_width + 1, "down": self.background_width + 1 })
			self.pipe_pos_y_list.append({ "up": pipe_pos_y_offset - self.pipe_height, "down": pipe_pos_y_offset + self.pipe_pos_y_gap })
		if self.pipe_pos_x_list[0]["up"] < -self.pipe_width:
			self.pipe_pos_x_list.pop(0)
			self.pipe_pos_y_list.pop(0)
