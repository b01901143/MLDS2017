from parse import *

read_data("./data/raw/Holmes/", small = True)

#path setting
in_file_path = "./data/raw/cut/big.txt"
out_file_path = "./data/sets/cut/"

#in_file, out_file
in_file = open(in_file_path, "r")
train_file = open(out_file_path + "train.txt", "w")
valid_file = open(out_file_path + "valid.txt", "w")
test_file = open(out_file_path + "test.txt", "w")

#in_data
in_data = in_file.readlines()
train_data = in_data[:len(in_data)*4/5]
valid_data = in_data[len(in_data)*4/5:len(in_data)*9/10]
test_data = in_data[len(in_data)*9/10:]

#divide
for d in train_data:
	train_file.write(d)
for d in valid_data:
	valid_file.write(d)
for d in test_data:
	test_file.write(d)

