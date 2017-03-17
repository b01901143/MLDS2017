import numpy as np
import pandas as pd


def get_questions():
	
	questions = []

	# print files
	
	df = pd.read_csv("testing_data.csv")

	for item in df['question']:
		words = map(str,item.split())

		i = 0
		while i < len(words):
			words[i] = words[i].lower()
			j = 0
			while j < len(words[i]):
				if not words[i][j].isalpha() and words[i] != '_____':
					words[i] = words[i].replace(words[i][j], "")
				else:
					j += 1

			if words[i] == "":
				del words[i]

			else:
				i += 1
				

		questions.append(map(str,"<start> <start>".split()) + words + map(str,"<end> <end>".split()))

	for idx in range(0,len(questions)):
		for j in range(0,len(questions[idx])):
			if questions[idx][j] == '_____':
				questions[idx][j] = ' '
				questions[idx] = questions[idx][j-2 : j+3]
				break


		# print questions[-1]
	return questions

def get_options():
	
	df = pd.read_csv("testing_data.csv")

	options = []

	dat = map(list, df.values)


	for item in dat:
		options.append (item[2:])

		for i,word in enumerate(options[-1]):

			word = word.lower()
			j = 0
			while j < len(word):
				if not word[j].isalpha():
					word = word.replace(word[j], "")
				else:
					j += 1
			options[-1][i] = word


	return options



'''

example of usage

both are list of lists

############################
a = get_questions()
b = get_options()

for i in range(0,len(a)):
	print a[i]
	print b[i]
	raw_input()
############################

'''