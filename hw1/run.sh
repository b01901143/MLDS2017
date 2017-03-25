savezip = save.zip
wget -O savefile
unzip -u-o savefile
#get the saved model
python cloze_prediction.py 0 $1 $2 hand_in 
#0 means don't reparse training data,$1 is path to testing data , $2 is output data path
