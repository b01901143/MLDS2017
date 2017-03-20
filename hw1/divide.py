from parse import *

read_data("./Holmes_Training_Data/")

in_file = open("train.txt", "r")
out_path = "./data/"
train_file = open(out_path + "train.txt", "w")
valid_file = open(out_path + "valid.txt", "w")
test_file = open(out_path + "test.txt", "w")

in_data = in_file.readlines()
train_data = in_data[:len(in_data)*4/5]
valid_data = in_data[len(in_data)*4/5:len(in_data)*9/10]
test_data = in_data[len(in_data)*9/10:]

for d in train_data:
	train_file.write(d)

for d in valid_data:
	valid_file.write(d)

for d in test_data:
	test_file.write(d)
