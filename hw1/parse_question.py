import numpy as np
import pandas as pd

def get_questions(test_num_steps , test_path):
	questions = []
	#df = pd.read_csv("./data/test/testing_data.csv")
	df=pd.read_csv(test_path)
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
		padding = "<end> " * (test_num_steps//2)
		questions.append(map(str,padding.split()) + words[:] + map(str,padding.split()))
	for idx in range(0,len(questions)):
		for j in range(0,len(questions[idx])):
			if questions[idx][j] == '_____':
				questions[idx][j] = ' '
				questions[idx] = questions[idx][j-test_num_steps//2 : j+test_num_steps//2 +2]
				break
	return questions

def get_options(test_path):
	#df = pd.read_csv("./data/test/testing_data.csv")
	df=pd.read_csv(test_path)
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
