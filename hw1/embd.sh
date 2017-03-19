wget -O embd.zip https://www.dropbox.com/s/s54agpg1k0yahc3/embd.zip?dl=1
unzip embd.zip 
cd ./embd
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..
