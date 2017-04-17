import tensorflow as tf
import numpy as np
from utility import *
from parameter import *

class Video_Caption_Generator():
    def __init__(self, video_size, n_words, hidden_size, batch_size, n_lstm_steps, video_step, caption_step, bias_init_vector=None):
        self.video_size = video_size
        self.n_words = n_words
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.video_step=video_step
        self.caption_step=caption_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, hidden_size], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([hidden_size]), name='bemb')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=False)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([video_size, hidden_size], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([hidden_size]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([hidden_size, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.video_step, self.video_size])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.video_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.caption_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.caption_step+1])

        video_flat = tf.reshape(video, [-1, self.video_size])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, hidden_size)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.hidden_size])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.hidden_size])
        probs = []
        loss = 0.0

        with tf.variable_scope("RNN"):

            ##############################  Encoding Stage ##################################
            for i in range(0, self.video_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:,i,:], state1)
                    print image_emb[:, i, :].get_shape()

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

            ############################# Decoding Stage ######################################
            for i in range(0, self.caption_step): ## Phase 2 => only generate captions
                #if i == 0:
                #    current_embed = tf.zeros([self.batch_size, self.hidden_size])
                #else:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

                labels = tf.expand_dims(caption[:, i+1], 1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat([indices, labels], 1)
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels)
                cross_entropy = cross_entropy * caption_mask[:,i]
                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
                loss = loss + current_loss

        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        
        # tf.get_variable_scope().reuse_variables()

        video = tf.placeholder(tf.float32, [1, self.video_step, self.video_size])

        video_flat = tf.reshape(video, [-1, self.video_size])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.video_step, self.hidden_size])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.hidden_size])

        generated_words = []

        probs = []
        embeds = []

        with tf.variable_scope("RNN"):

            for i in range(0, self.video_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

            for i in range(0, self.caption_step):
                tf.get_variable_scope().reuse_variables()

                if i == 0:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

                logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_words.append(max_prob_index)
                probs.append(logit_words)

                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                    current_embed = tf.expand_dims(current_embed, 0)

                embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds

def train():
    train_data, test_data = getInfo("training"), getInfo("testing")
    word_id, id_word = buildVocab(train_data["label_sentence"].values)
    print len(word_id)
    '''
    train_captions = train_data["Description"].values
    
    captions_list = list(train_captions) 
    captions = np.asarray(captions_list, dtype=np.object)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)

    np.save("./data/training/wordtoix", wordtoix)
    np.save('./data/training/ixtoword', ixtoword)
    np.save("./data/training/bias_init_vector", bias_init_vector)

    model = Video_Caption_Generator(
            video_size=video_size,
            n_words=len(wordtoix),
            hidden_size=hidden_size,
            batch_size=batch_size,
            n_lstm_steps=video_step,
            video_step=video_step,
            caption_step=caption_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()

    #new_saver = tf.train.Saver()
    #new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []

    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []

        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]

        current_train_data = train_data.groupby('feat_path').apply(lambda x: x.irow(np.random.choice(len(x))))
        current_train_data = current_train_data.reset_index(drop=True)

        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            start_time = time.time()

            current_batch = current_train_data[start:end]
            current_videos = current_batch['feat_path'].values

            current_feats = np.zeros((batch_size, video_step, video_size))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)

            current_video_masks = np.zeros((batch_size, video_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = current_batch["Description"].values
            current_captions = map(lambda x: '<bos> ' + x, current_captions)
            current_captions = map(lambda x: x.replace('.', ''), current_captions)
            current_captions = map(lambda x: x.replace(',', ''), current_captions)
            current_captions = map(lambda x: x.replace('"', ''), current_captions)
            current_captions = map(lambda x: x.replace('\n', ''), current_captions)
            current_captions = map(lambda x: x.replace('?', ''), current_captions)
            current_captions = map(lambda x: x.replace('!', ''), current_captions)
            current_captions = map(lambda x: x.replace('\\', ''), current_captions)
            current_captions = map(lambda x: x.replace('/', ''), current_captions)

            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < caption_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(caption_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)
            # print current_caption_ind
            current_caption_matrix = np.zeros( (len(current_caption_ind), caption_step), dtype=np.int )
            for i in range(len(current_caption_ind)):
                current_caption_matrix[ i ,:len(current_caption_ind[i]) ] = current_caption_ind[i]
            # print current_caption_matrix
            # current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=caption_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)

            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video:current_feats,
                tf_caption: current_caption_matrix
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            loss_to_draw_epoch.append(loss_val)

            print 'idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time))
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        # draw loss curve every epoch
        #loss_to_draw.append(np.mean(loss_to_draw_epoch))
        #plt_save_dir = "./loss_imgs"
        #plt_save_img_name = str(epoch) + '.png'
        #plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        #plt.grid(True)
        #plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

        if np.mod(epoch, 10) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

    loss_fd.close()

def test(model_path='./models/model-100'):
    test_data = get_data(video_test_feat_path, video_test_data_path)
    test_videos = test_data['video_path'].unique()

    ixtoword = pd.Series(np.load('./training_data/ixtoword.npy').tolist())

    bias_init_vector = np.load('./training_data/bias_init_vector.npy')

    model = Video_Caption_Generator(
            video_size=video_size,
            n_words=len(ixtoword),
            hidden_size=hidden_size,
            batch_size=batch_size,
            n_lstm_steps=video_step,
            video_step=video_step,
            caption_step=caption_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open('S2VT_results.txt', 'w')
    for idx, video_feat_path in enumerate(test_videos):
        print idx, video_feat_path

        video_feat = np.load(video_feat_path)[None,...]
        #video_feat = np.load(video_feat_path)
        #video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        if video_feat.shape[1] == video_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:
            continue
            #shape_templete = np.zeros(shape=(1, video_step, 4096), dtype=float )
            #shape_templete[:video_feat.shape[0], :video_feat.shape[1], :video_feat.shape[2]] = video_feat
            #video_feat = shape_templete
            #video_mask = np.ones((video_feat.shape[0], video_step))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        print generated_sentence,'\n'
        test_output_txt_fd.write(video_feat_path + '\n')
        test_output_txt_fd.write(generated_sentence + '\n\n')
'''
# def train1():
#     # get_data('training_data/feat/')
#     # exit()
#     train_data = get_data(video_train_data_path, video_train_feat_path)
#     train_captions = train_data["Description"].values
#     test_data = get_data(video_test_data_path, video_test_feat_path)
#     test_captions = test_data["Description"].values

train()
