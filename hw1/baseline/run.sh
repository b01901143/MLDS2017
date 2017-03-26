wget -O save.zip https://www.dropbox.com/s/xviu2iqbukhmn4f/save.zip?dl=1
wget -O data.zip https://www.dropbox.com/s/heacx5oy3ae43pl/data.zip?dl=1
unzip -u-o save.zip
unzip -u-o data.zip
#get the saved model
python test.py $1 $2  
#0 means don't reparse training data,$1 is path to testing data , $2 is output data path
