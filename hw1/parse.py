import numpy as np
import tensorflow as tf

from os import listdir
from os.path import isfile, join

mypath = 'Holmes_Training_Data/'

#not handling Mr. ,Mrs. !!!!
#not handling I'm !!!!
#all lower case
#remove all marks


def read_data(mypath):
	num = 0
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	sentences = []

	# print files
	for f in files[:]:
		
		f_in = open( mypath + f, 'r')
		lines = f_in.readlines()
		started = False
		head_count = 999 # meaningless in start

		sentence = str()

		for i,line in enumerate(lines):  # only use 1/5 of every book

			if started and head_count>0: # skip the first 70 lines, may be content
				head_count -= 1

			if head_count == 0:

				if i > stop_line:
					break
				
				if len(line) < 3:
					continue

				# print line

				# words = map(str,line.split())
				
				if line[-3] not in ['.', '!', '?',"\""] and len(lines[i+1]) < 3:
					continue
				sentence += line[:-2] + ' '


				


  			if line[0:21] == '*END*THE SMALL PRINT!': ## one file missing!!!!*END*THE SMALL PRINT!
  				
  				started = True
				length_eff = len(lines) - i

				head_count = length_eff // 5 * 2

				stop_line = len(lines) - length_eff // 5 * 2

  		count = 0

		# print sentence
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
						sentences.append(map(str,"<start> <start>".split()) + words[pivot_pre : i+1] + map(str,"<end> <end>".split()))
						count += 1
						pivot_pre = i+1

				i += 1
				if i >= len(words):
					break
			# print words[i-1]
			# raw_input()
		print f, count
		print len(sentences)
		return sentences



	chopped = []
	f_out = open("cut.txt", 'w') 
	# f_out_5 = open("chopped_5.txt", 'w') 
	for st in sentences:
		for wd in st:
			f_out.write(wd + ' ')

		# for idx in range(0,len(st)-4):
		# 	chopped.append(st[idx:idx+5])

		# 	for wd in st[idx:idx+5]:
		# 		f_out_5.write(wd + ' ')
		# 	f_out_5.write('\n')
		f_out.write('\n')
	print sentences
	print len(sentences)
	raw_input()


read_data(mypath)

