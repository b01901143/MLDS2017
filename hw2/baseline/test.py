import json
import pandas as pd
import numpy as np
import tensorflow as tf
from utility import *
from parameter import *
from structure import *
from bleu import *

test_model_version = 420
test_model_path = model_dir + "-" + str(test_model_version)
testing_id_path = sys.argv[1]
feature_path = sys.argv[2]
def test():
    #prepare data
    train_data, test_data = getInfo(train_info_path), getInfo(test_info_path)
    train_label = [ json.load(open(train_label_dir + path)) for path in train_data["label_path"].values ]
    test_label_split = [ json.load(open(test_label_dir + path)) for path in test_data["label_path"].values ]
    test_label_json = json.load(open(test_label_all))
    test_label =  {label['id'] : label['caption'] for label in test_label_json}  
    if Embd_flag is True:
		_, id_word, init_bias_vector,embd = buildEmbd(train_label + test_label_split)
    else:
		_, id_word, init_bias_vector = buildVocab(train_label + test_label_split)	
    #initialize model
    model = VideoCaptionGenerator(
            video_size=video_size,
            video_step=video_step,
            caption_size=caption_size,
            caption_step=caption_step,
            hidden_size=hidden_size,
            batch_size=batch_size,
            output_keep_prob=output_keep_prob,
            init_bias_vector=init_bias_vector,
			pretrained_embd = embd
        )
    #build model
    tf_video_array, tf_video_array_mask, tf_caption_array_id = model.buildGenerator()
    #build session, saver
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    #restore variables
    saver.restore(session, test_model_path)
    #run testing

    f_test_id = open(testing_id_path)
    output_list = [] 
    for index, feat_path in enumerate(f_test_id.readlines()):
        	
        print "VideoID: " + str(index) + " Path: " + feat_path.strip('\n')
        video_array = np.load(feature_path+feat_path.strip('\n')+'.npy')[None,...]                    # adjusted for handout
        video_array_mask = np.ones((video_array.shape[0], video_array.shape[1]))
        #caption_array
        fetch_dict = {
            "caption_array_id":tf_caption_array_id
        }
        feed_dict = {
            tf_video_array:video_array,
            tf_video_array_mask:video_array_mask
        }
        track_dict = session.run(fetch_dict, feed_dict)
        caption_array_id = track_dict["caption_array_id"]
        caption_array = [ id_word[idx].encode('utf-8') for arr in caption_array_id for idx in arr ]
        caption_string= arr2str(caption_array)   
        print caption_string
        _caption_dic = {'caption': caption_string.strip(' \n'),'id': feat_path.strip('\n') }
        output_list.append(_caption_dic)
    f_out = open('./output.json','w')
    json.dump(output_list,f_out, indent=4)	
    f_out.close()

    #print bleu
if __name__ == "__main__":
    test()
