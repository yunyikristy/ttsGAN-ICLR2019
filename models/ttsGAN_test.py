import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,LSTMCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from util.ops import shape_list
from .helpers import TestHelper, TrainingHelper
from .networks import encoder_cbhg, post_cbhg, prenet, reference_encoder
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper, ZoneoutWrapper
from .multihead_attention import MultiheadAttention


class ttsGAN_test():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, reference_mel=None):
    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      is_teacher_force_generating = mel_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings
      embedding_table = tf.get_variable(
        'text_embedding', [len(symbols), 256], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)           # [N, T_in, 256]
      
      if hp.use_gst:
        #Global style tokens (GST)
        gst_tokens = tf.get_variable(
          'style_tokens', [hp.num_gst, 256 // hp.num_heads], dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=0.5))
        self.gst_tokens = gst_tokens
 
      # Encoder
      prenet_outputs = prenet(embedded_inputs, is_training)                       # [N, T_in, 128]
      encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training)  # [N, T_in, 256]
      
      if is_training:
        reference_mel = mel_targets

      if reference_mel is not None:
        # Reference encoder
        refnet_outputs = reference_encoder(
          reference_mel, 
          filters=[32, 32, 64, 64, 128, 128], 
          kernel_size=(3,3),
          strides=(2,2),
          encoder_cell=GRUCell(128),
          is_training=is_training)                                                 # [N, 128]
        self.refnet_outputs = refnet_outputs                                       

        if hp.use_gst:
          # Style attention
          style_attention = MultiheadAttention(
            tf.tanh(tf.expand_dims(refnet_outputs, axis=1)),                                   # [N, 1, 128]
            tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size,1,1]),            # [N, hp.num_gst, 256/hp.num_heads]   
            num_heads=hp.num_heads,
            num_units=128,
            attention_type=hp.style_att_type)

          # Apply tanh to compress both encoder state and style embedding to the same scale.
          style_embeddings = style_attention.multi_head_attention()                   # [N, 1, 256]
        else:
          style_embeddings = tf.expand_dims(refnet_outputs, axis=1)                   # [N, 1, 128]
      else:
        print("Use random weight for GST.")
        random_weights = tf.random_uniform([hp.num_heads, hp.num_gst], maxval=1.0, dtype=tf.float32)
        random_weights = tf.nn.softmax(random_weights, name="random_weights")
        style_embeddings = tf.matmul(random_weights, tf.nn.tanh(gst_tokens))
        style_embeddings = tf.reshape(style_embeddings, [1, 1] + [hp.num_heads * gst_tokens.get_shape().as_list()[1]])

      # Add style embedding to every text encoder state
      style_embeddings = tf.tile(style_embeddings, [1, shape_list(encoder_outputs)[1], 1]) # [N, T_in, 128]
      encoder_outputs = tf.concat([encoder_outputs, style_embeddings], axis=-1)

      # Attention
      attention_cell = AttentionWrapper(
        DecoderPrenetWrapper(GRUCell(256), is_training),
        BahdanauAttention(256, encoder_outputs, memory_sequence_length=input_lengths),
        alignment_history=True,
        output_attention=False)                                                  # [N, T_in, 256]

      # Concatenate attention context vector and RNN cell output.
      concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)              

      # Decoder (layers specified bottom to top):
      decoder_cell = MultiRNNCell([
          OutputProjectionWrapper(concat_cell, 256),
          ResidualWrapper(ZoneoutWrapper(LSTMCell(256), 0.1)),
          ResidualWrapper(ZoneoutWrapper(LSTMCell(256), 0.1))
        ], state_is_tuple=True)                                                  # [N, T_in, 256]

      # Project onto r mel spectrograms (predict r outputs at each RNN step):
      output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.outputs_per_step)
      decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      if is_training or is_teacher_force_generating:
        helper = TrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
      else:
        helper = TestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

      (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(output_cell, helper, decoder_init_state),
        maximum_iterations=hp.max_iters)                                        # [N, T_out/r, M*r]

      # Reshape outputs to be one output per entry
      mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hp.num_mels]) # [N, T_out, M]

      # Add post-processing CBHG:
      post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training)           # [N, T_out, 256]
      linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)               # [N, T_out, F]

     # # Grab alignments from the final decoder state:
     # alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_outputs = mel_outputs
      self.encoder_outputs = encoder_outputs
      self.style_embeddings = style_embeddings
      self.linear_outputs = linear_outputs
     # self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      self.reference_mel = reference_mel
       


