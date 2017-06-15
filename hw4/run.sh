wget -O model.zip https://www.dropbox.com/s/73xittzilvuq5lm/model.zip?dl=1
mkdir data
unzip model.zip -d data/
sh test_RL.sh
sh test_S2S.sh
