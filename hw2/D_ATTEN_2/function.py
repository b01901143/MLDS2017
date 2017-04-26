import copy
import tensorflow as tf
from six.moves import xrange
from six.moves import zip
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

def top_wrapper(
        encoder_inputs,
        decoder_inputs,
        cell, 
        source_size, 
        target_size, 
        embedding_size,
        output_projection, 
        feed_previous, 
    ):
    with variable_scope.variable_scope(None or "embedding_attention_seq2seq", dtype=tf.float32) as scope:
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = core_rnn_cell.InputProjectionWrapper(encoder_cell, embedding_size)
        encoder_outputs, encoder_state = core_rnn.static_rnn(encoder_cell, encoder_inputs, tf.float32)
        top_states = [ 
            array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
        ]
        attention_states = array_ops.concat(top_states, 1)
        if isinstance(feed_previous, bool):
            return tf.contrib.legacy_seq2seq.embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                cell,
                target_size,
                embedding_size,
                num_heads=1,
                output_size=None,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=False
            )
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
                outputs, state = tf.contrib.legacy_seq2seq.embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=1,
                    output_size=None,
                    output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False,
                    initial_state_attention=False
                )
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list
        outputs_and_state = control_flow_ops.cond(
            feed_previous,
            lambda: decoder(True),
            lambda: decoder(False)
        )
        outputs_len = len(decoder_inputs)
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state, flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state
