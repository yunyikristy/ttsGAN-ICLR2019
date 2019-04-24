import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,LSTMCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from util.ops import shape_list
from .helpers import TestHelper, TrainingHelper
from .networks import encoder_cbhg, post_cbhg, prenet, reference_encoder, discriminator
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper, ZoneoutWrapper
from .multihead_attention import MultiheadAttention


class ttsGAN():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, mel_targets_pos=None, linear_targets_pos=None, mel_targets_neg=None, linear_targets_neg=None, labels_pos=None, labels_neg=None, reference_mel_pos=None, reference_mel_neg=None):
    
    is_training = linear_targets_pos is not None
    is_teacher_force_generating = mel_targets_pos is not None
    batch_size = tf.shape(inputs)[0]
    hp = self._hparams
    
    ## Text Encoding scope
    with tf.variable_scope('text_encoder',reuse=tf.AUTO_REUSE) as scope:
      # Initialize Text Embeddings
      embedding_table = tf.get_variable(
        'text_embedding', [len(symbols), 256], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)           # [N, T_in, 256]
        
      # Text Encoder
      prenet_outputs = prenet(embedded_inputs, is_training)                       # [N, T_in, 128]
      encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training)  # [N, T_in, 256]
      
      content_inputs = encoder_outputs

     ## Reference Encoding Scope
    with tf.variable_scope('audio_encoder', reuse=tf.AUTO_REUSE) as scope:
      
      if hp.use_gst:
         #Global style tokens (GST)
         gst_tokens = tf.get_variable(
           'style_tokens', [hp.num_gst, 256 // hp.num_heads], dtype=tf.float32,
           initializer=tf.truncated_normal_initializer(stddev=0.5))
         self.gst_tokens = gst_tokens

      if is_training:
        
        reference_mel_pos = mel_targets_pos
        reference_mel_neg = mel_targets_neg

      if reference_mel_pos is not None:
        # Reference encoder
        refnet_outputs_pos = reference_encoder(
          reference_mel_pos, 
          filters=[32, 32, 64, 64, 128, 128], 
          kernel_size=(3,3),
          strides=(2,2),
          encoder_cell=GRUCell(128),
          is_training=is_training)                                                 # [n, 128]
        self.refnet_outputs_pos = refnet_outputs_pos                                       
     
        refnet_outputs_neg = reference_encoder(
          reference_mel_neg, 
          filters=[32, 32, 64, 64, 128, 128], 
          kernel_size=(3,3),
          strides=(2,2),
          encoder_cell=GRUCell(128),
          is_training=is_training)                                                 # [n, 128]
        self.refnet_outputs_neg = refnet_outputs_neg                                       
        # Extract style features 
        ref_style = style_encoder(
          reference_mel_neg,
          filters=[32, 32, 64, 64],
          kernel_size=(3, 3),
          strides=(2,2),
          is_training=False)
        self.ref_style = ref_style
        
         
        if hp.use_gst:
          # Multi-head attention
          style_attention_pos = MultiheadAttention(
            tf.tanh(tf.expand_dims(refnet_outputs_pos, axis=1)),                                   # [N, 1, 128]
            tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size,1,1]),            # [N, hp.num_gst, 256/hp.num_heads]   
            num_heads=hp.num_heads,
            num_units=128,
            attention_type=hp.style_att_type)

          style_attention_neg = MultiheadAttention(
            tf.tanh(tf.expand_dims(refnet_outputs_neg, axis=1)),                                   # [N, 1, 128]
            tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size,1,1]),            # [N, hp.num_gst, 256/hp.num_heads]   
            num_heads=hp.num_heads,
            num_units=128,
            attention_type=hp.style_att_type)

          # Apply tanh to compress both encoder state and style embedding to the same scale.
          
          style_embeddings_pos = style_attention_pos.multi_head_attention()                   # [N, 1, 256]
          style_embeddings_neg = style_attention_neg.multi_head_attention()                   # [N, 1, 256]
 
        else:
          style_embeddings_pos = tf.expand_dims(refnet_outputs_pos, axis=1)                   # [N, 1, 128]
          style_embeddings_neg = tf.expand_dims(refnet_outputs_neg, axis=1)
      else:
        print("Use random weight for GST.")
        
      # Add style embedding to every text encoder state
      ## tile style embeddings such that it could matched with text sequence shape,
      ## format: _content_style
      style_embeddings_pos = tf.tile(style_embeddings_pos, [1, shape_list(encoder_outputs)[1], 1]) # [N, T_in, 128]
      style_embeddings_neg = tf.tile(style_embeddings_neg, [1, shape_list(encoder_outputs)[1], 1]) # [N, T_in, 128]
      ## purmute four encoder outputs, e.g. pos2pos is positive content wieh positive style, pos2neg is postive content wity
      ## negtive style.
      encoder_outputs_pos = tf.concat([encoder_outputs, style_embeddings_pos], axis=-1)
      encoder_outputs_neg = tf.concat([encoder_outputs, style_embeddings_neg], axis=-1)
   
    # Decoding scope
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
      # RNN Attention
      attention_cell_pos = AttentionWrapper(
        DecoderPrenetWrapper(GRUCell(256), is_training),
        BahdanauAttention(256, encoder_outputs_pos, memory_sequence_length=input_lengths),
        alignment_history=True,
        output_attention=False)                                                  # [N, T_in, 256]

      attention_cell_neg = AttentionWrapper(
        DecoderPrenetWrapper(GRUCell(256), is_training),
        BahdanauAttention(256, encoder_outputs_neg, memory_sequence_length=input_lengths),
        alignment_history=True,
        output_attention=False)                                                  # [N, T_in, 256]
      
      # Concatenate attention context vector and RNN cell output.
      concat_cell_pos = ConcatOutputAndAttentionWrapper(attention_cell_pos)              
      concat_cell_neg = ConcatOutputAndAttentionWrapper(attention_cell_neg)              
      
          
      # Decoder (layers specified bottom to top):
      decoder_cell_pos = MultiRNNCell([
          OutputProjectionWrapper(concat_cell_pos, 256),
          ResidualWrapper(ZoneoutWrapper(LSTMCell(256), 0.1)),
          ResidualWrapper(ZoneoutWrapper(LSTMCell(256), 0.1))
        ], state_is_tuple=True)                                                  # [N, T_in, 256]
      
      decoder_cell_neg = MultiRNNCell([
          OutputProjectionWrapper(concat_cell_neg, 256),
          ResidualWrapper(ZoneoutWrapper(LSTMCell(256), 0.1)),
          ResidualWrapper(ZoneoutWrapper(LSTMCell(256), 0.1))
        ], state_is_tuple=True)                                                  # [N, T_in, 256]

       
      # Project onto r mel spectrograms (predict r outputs at each RNN step):
      output_cell_pos = OutputProjectionWrapper(decoder_cell_pos, hp.num_mels * hp.outputs_per_step)
      decoder_init_state_pos = output_cell_pos.zero_state(batch_size=batch_size, dtype=tf.float32)

      output_cell_neg = OutputProjectionWrapper(decoder_cell_neg, hp.num_mels * hp.outputs_per_step)
      decoder_init_state_neg = output_cell_neg.zero_state(batch_size=batch_size, dtype=tf.float32)

      
      if is_training or is_teacher_force_generating:
        helper_pos = TacoTrainingHelper(inputs, mel_targets_pos, hp.num_mels, hp.outputs_per_step)
        helper_neg = TacoTrainingHelper(inputs, mel_targets_neg, hp.num_mels, hp.outputs_per_step)
        
      else:
        helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

      (decoder_outputs_pos, _), final_decoder_state_pos, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(output_cell_pos, helper_pos, decoder_init_state_pos),
        maximum_iterations=hp.max_iters)                                        # [N, T_out/r, M*r]
      
      (decoder_outputs_neg, _), final_decoder_state_neg, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(output_cell_neg, helper_neg, decoder_init_state_neg),
        maximum_iterations=hp.max_iters)                                        # [N, T_out/r, M*r]
       
      # Reshape outputs to be one output per entry
      
      mel_outputs_pos = tf.reshape(decoder_outputs_pos, [batch_size, -1, hp.num_mels]) # [N, T_out, M]
      mel_outputs_neg = tf.reshape(decoder_outputs_neg, [batch_size, -1, hp.num_mels]) # [N, T_out, M]
      
      # Add post-processing CBHG:
      post_outputs_pos = post_cbhg(mel_outputs_pos, hp.num_mels, is_training)           # [N, T_out, 256]
      linear_outputs_pos = tf.layers.dense(post_outputs_pos, hp.num_freq)               # [N, T_out, F]
      
      post_outputs_neg = post_cbhg(mel_outputs_neg, hp.num_mels, is_training)           # [N, T_out, 256]
      linear_outputs_neg = tf.layers.dense(post_outputs_neg, hp.num_freq)               # [N, T_out, F]
 
      ## Grab alignments from the final decoder state:
      alignments_pos = tf.transpose(final_decoder_state_pos[0].alignment_history.stack(), [1, 2, 0])
      alignments_neg = tf.transpose(final_decoder_state_neg[0].alignment_history.stack(), [1, 2, 0])

      # Extract style features for fake sample
      rec_style = style_encoder(
        mel_outputs_neg,
        filters=[32, 32, 64, 64],
        kernel_size=(3,3),
        strides=(2,2),
        is_training=False)
      self.rec_style = rec_style

    # Discriminator scope
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
      self.real_logit = discriminator(content_inputs, reference_mel_pos, is_training=is_training) 
      self.fake_logit_pos = discriminator(content_inputs, mel_outputs_pos, is_training=is_training)
      self.fake_logit_neg = discriminator(content_inputs, mel_outputs_neg, is_training=is_training)
    
    self.inputs = inputs
    self.input_lengths = input_lengths
    self.mel_outputs_pos = mel_outputs_pos
    self.mel_outputs_neg = mel_outputs_neg
    
    self.encoder_outputs = encoder_outputs
              
    self.style_embeddings_pos = style_embeddings_pos
    self.style_embeddings_neg = style_embeddings_neg
    
    self.linear_outputs_pos = linear_outputs_pos
    self.linear_outputs_neg = linear_outputs_neg
    
    self.alignments_pos = alignments_pos
    self.alignments_neg = alignments_neg
    self.mel_targets_pos = mel_targets_pos
    self.mel_targets_neg = mel_targets_neg
    self.linear_targets_pos = linear_targets_pos
    self.linear_targets_neg = linear_targets_neg
    self.reference_mel_pos = reference_mel_pos
    self.reference_mel_neg = reference_mel_neg
    log('Initialized Tacotron model. Dimensions: ')
    log('text embedding:          %d' % embedded_inputs.shape[-1])
    #log(' negative text embedding:           %d' % embedded_inputs_neg.shape[-1])
    #log('  style embedding:         %d' % style_embeddings.shape[-1])
    #log('  prenet out:              %d' % prenet_outputs.shape[-1])
    #log('  encoder out:             %d' % encoder_outputs.shape[-1])
    #log('  attention out:           %d' % attention_cell.output_size)
    #log('  concat attn & out:       %d' % concat_cell.output_size)
    #log('  decoder cell out:        %d' % decoder_cell.output_size)
    #log('  decoder out (%d frames):  %d' % (hp.outputs_per_step, decoder_outputs.shape[-1]))
    #log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])
    #log('  postnet out:             %d' % post_outputs.shape[-1])
    #log('  linear out:              %d' % linear_outputs.shape[-1])


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      hp = self._hparams
      batch_size = 16.
            
      ## Adversarial Game
      ## GAN loss
      real_d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=real_logit, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
      real_d_loss = tf.reduce_mean(real_d_loss)
      fake_d_loss_pos = tf.nn.softmax_cross_entropy_with_logits(logits=fake_logit_pos, labels=tf.constant([[0.0, 1.0, 0.0]]))
      fake_d_loss_pos = tf.reduce_mean(fake_d_loss_pos)
      fake_d_loss_neg = tf.nn.softmax_cross_entropy_with_logits(logits=fake_logit_neg, labels=tf.constant([[0.0, 0.0, 1.0]]))
      fake_d_loss_neg = tf.reduce_mean(fake_d_loss_neg)

      fake_g_loss_pos = tf.nn.softmax_cross_entropy_with_logits(logits=fake_logit_pos, labels=tf.constant([[1.0, 0.0, 0.0]]))
      fake_g_loss_pos = tf.reduce_mean(fake_g_loss_pos)
      fake_g_loss_neg = tf.nn.softmax_cross_entropy_with_logits(logits=fake_logit_neg, labels=tf.constant([[1.0, 0.0, 0.0]]))
      fake_g_loss_neg = tf.reduce_mean(fake_g_loss_neg)

      self.d_loss = real_d_loss + (fake_d_loss_pos + fake_d_loss_neg)/2.
      self.g_loss = (gen_loss_pos + gen_loss_neg)/2.

      ## Collaboratvie Game
      ## Reconstruction loss in original space
      self.mel_loss_pos = tf.reduce_mean(tf.abs(self.mel_targets_pos - self.mel_outputs_pos))
      self.linear_loss_pos = tf.reduce_mean(tf.abs(self.linear_targets_pos - self.linear_outputs_pos))

      ## Reconstruction loss in latent space
      neg_target_logit = c_net(self.refnet_outputs_neg, is_training)
      pos_target_logit = c_net(self.refnet_outputs_pos, is_training)

      refnet_rec_neg = reference_encoder(
          self.mel_output_neg, 
          filters=[32, 32, 64, 64, 128, 128], 
          kernel_size=(3,3),
          strides=(2,2),
          encoder_cell=GRUCell(128),
          is_training=is_training)                                                 # [n, 128]

      refnet_rec_pos = reference_encoder(
          self.mel_output_pos, 
          filters=[32, 32, 64, 64, 128, 128], 
          kernel_size=(3,3),
          strides=(2,2),
          encoder_cell=GRUCell(128),
          is_training=is_training)                                                 # [n, 128]

      neg_rec_logit = c_net(refnet_rec_neg, is_training)
      pos_rec_logit = c_net(refnet_rec_pos, is_training)

      
        
      self.neg_target_c_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_neg, logits=neg_target_logit)
      self.neg_target_c_loss = tf.reduce_mean(self.neg_target_c_loss)
      self.neg_rec_c_loss = tf.nn.softmax_cross_ebtropy_with_logits(labels=self.labels_neg, logits=neg_rec_logit)
      self.neg_rec_c_loss = tf.reduce_mean(self.neg_rec_c_loss)

      self.pos_target_c_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_pos, logits=pos_target_logit)
      self.pos_target_c_loss = tf.reduce_mean(self.pos_target_c_loss)
      self.pos_rec_c_loss = tf.nn.softmax_cross_ebtropy_with_logits(labels=self.labels_pos, logits=pos_rec_logit)
      self.pos_rec_c_loss = tf.reduce_mean(self.pos_rec_c_loss)

      self.rec_loss = self.mel_loss_pos + self.linear_loss_pos + self.neg_target_c_loss + self.neg_rec_c_loss + self.pos_target_c_loss + self.pos_rec_c_loss

      ## compute style loss
      STYLE_LAYERS=[]
      for i in range(4):
        name = 'conv2d_%d' %i
        STYLE_LAYERS.append(name)
      style_features = {}

      for layer in STYLE_LAYERS:
        feature = self.ref_style[layer]
        feature = tf.reshape(feature, [-1, feature.shape[3]])
        dim = feature.shape[1].value
        size = batch_size * dim
        gram = tf.matmul(tf.transpose(feature), feature) / size
        style_features[layer] = gram

      style_features_rec = {}
      for layer in STYLE_LAYERS:
        feature = self.rec_style[layer]
        feature = tf.reshape(feature, [-1, feature.shape[3]])
        dim = feature.shape[1].value
        size = batch_size * dim
        gram = tf.matmul(tf.transpose(feature), feature) / size
        style_features_rec[layer] = gram

      style_loss=0
      for layer in STYLE_LAYERS:
        rec_feature = style_features_rec[layer]
        dim = rec_feature.shape[1].value
        size = batch_size * dim
        style_loss += tf.nn.l2_loss(style_features[layer] - rec_feature) / size

      self.style_loss = tf.reduce_mean(style_loss)
      self.loss = 10. * self.rec_loss + adv_loss +  0.1 * self.style_loss


  def add_optimizer(self, global_step):
    all_vars = tf.trainable_variables()
 
    d_vars = [var for var in all_vars if
              var.name.startswith('model/discriminator')]
    g_vars = [var for var in all_vars if
              var.name.startswith('model/generator')]
    text_e_vars = [var for var in all_vars if 
              var.name.startswith('model/text_')]
    audio_e_vars = [var for var in all_vars if
              var.name.startswith('model/audio_')]

    with tf.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)

      g_optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      g_gradients, g_variables = zip(*g_optimizer.compute_gradients(loss = self.rec_loss + self.g_loss, var_list = text_e_vars + audio_e_vars +  g_vars))
      self.g_gradients = g_gradients
      clipped_g_gradients, _ = tf.clip_by_global_norm(g_gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.g_optimize = g_optimizer.apply_gradients(zip(clipped_g_gradients, g_variables),
          global_step=global_step)

      d_optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      d_gradients, d_variables = zip(*d_optimizer.compute_gradients(self.d_loss, var_list = d_vars))
      self.d_gradients = d_gradients
      clipped_d_gradients, _ = tf.clip_by_global_norm(d_gradients, 1.0)

      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.d_optimize = d_optimizer.apply_gradients(zip(clipped_d_gradients, d_variables),
          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
