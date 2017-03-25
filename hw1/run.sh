savezip = save.zip
wget -O savefile
unzip -u-o savefile
#get the saved model
python cloze_prediction.py $1 $2 
#$1 is path to testing data , $2 is output data path
