import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join

def read_data(mypath, small = False):
	num = 0
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	train_data = []
	labels = []
	for f in files[:]:
		f_in = open( mypath + f, 'r')
		lines = f_in.readlines()
		started = False
		head_count = 999
		sentence = str()
		for i,line in enumerate(lines):
			if started and head_count>0:
				head_count -= 1
			if head_count == 0:
				if i > stop_line:
					break
				if len(line) < 3:
					continue
				if line[-3] not in ['.', '!', '?',"\""] and len(lines[i+1]) < 3:
					continue
				sentence += line[:-2] + ' '
  			if line[0:21] == '*END*THE SMALL PRINT!':
  				started = True
				length_eff = len(lines) - i
				head_count = length_eff // 5 * 2
				stop_line = len(lines) - length_eff // 5 * 2
				if small:
					head_count = length_eff // 35 * 17
					stop_line = len(lines) - length_eff // 35 * 17
  		count = 0
		words = map(str,sentence.split())
		first_sentence = True
		pivot_pre = 0
		i = 0
		while i < len(words):
			words[i] = words[i].lower()
			j = 0
			while j < len(words[i]):
				if not words[i][j].isalpha() and words[i][j] not in ['.', '!', '?']:
					words[i] = words[i].replace(words[i][j], "")
				else:
					j += 1
			if words[i] == "":
				del words[i]
			else:			
				if words[i][-1] in ['.', '!', '?']:
					if first_sentence:
						first_sentence = False
						pivot_pre = i+1
					else:
						words[i] = words[i].replace(words[i][-1], "")
						if not words[pivot_pre].isalpha():
							continue
						if not i - pivot_pre > 4:
							pivot_pre = i+1 # !!!!!!!!!!!!!
							continue
						temp = (map(str,"<start> <start>".split()) + words[pivot_pre : i+1] + map(str,"<end> <end>".split()))

						chop_idx = np.random.randint( len(temp)-5 , size=len(temp)//5)
						for idc in chop_idx:
							train_data.append(temp[idc :idc+5])
							labels.append(temp[idc+1 :idc+6])
							count += 1
						pivot_pre = i+1
				i += 1
				if i >= len(words):
					break
	return train_data, labels
