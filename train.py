import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import traceback

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit




def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  ## input path is lists of both postive path and negtiva path
  input_path_pos = os.path.join(args.base_dir, args.input_pos)
  input_path_neg = os.path.join(args.base_dir, args.input_neg)
  
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading positive training data from: %s' % input_path_pos)
  log('Loading negative training data from: %s' % input_path_neg)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  # Set up DataFeeder:
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path_pos, input_path_neg, hparams)

  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = create_model(args.model, hparams)
    model.initialize(feeder.inputs_pos, feeder.input_lengths_pos, feeder.mel_targets_pos, feeder.linear_targets_pos, feeder.mel_targets_neg, feeder.linear_targets_neg, feeder.labels_pos, feeder.labels_neg)
    model.add_loss()
    model.add_optimizer(global_step)
    

  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  # Train!
  with tf.Session() as sess:
    try:
      #summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      if args.restore_step:
        # Restore from a checkpoint if the user requested it.
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
      else:
        log('Starting new training run at commit: %s' % commit, slack=True)

      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()
        # train d
        sess.run(model.d_optimize)
        # train g
        step, rec_loss, style_loss, d_loss, g_loss, _ = sess.run([global_step, model.rec_loss, model.style_loss,  model.d_loss, model.g_loss, model.g_optimize])
        time_window.append(time.time() - start_time)
        message = 'Step %-7d [%.03f sec/step, rec_loss=%.05f, style_loss=%.05f, d_loss=%.05f, g_loss=%.05f]' % (
          step, time_window.average, rec_loss, style_loss, d_loss, g_loss)
        log(message, slack=(step % args.checkpoint_interval == 0))
         
        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          log('Saving audio and alignment...')
          input_seq, spectrogram_pos, spectrogram_neg, alignment_pos, alignment_neg = sess.run([
            model.inputs[0], model.linear_outputs_pos[0], model.linear_outputs_neg[0], model.alignments_pos[0], model.alignments_neg[0]])
          
          waveform_pos = audio.inv_spectrogram(spectrogram_pos.T)
          waveform_neg = audio.inv_spectrogram(spectrogram_neg.T)
          audio.save_wav(waveform_pos, os.path.join(log_dir, 'step-%d-audio_pos.wav' % step))
          audio.save_wav(waveform_neg, os.path.join(log_dir, 'step-%d-audio_neg.wav' % step))
          plot.plot_alignment(alignment_pos, os.path.join(log_dir, 'step-%d-align_pos.png' % step), 
            info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, rec_loss))
          plot.plot_alignment(alignment_neg, os.path.join(log_dir, 'step-%d-align_neg.png' % step),
            info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, rec_loss))
          log('Input: %s' % sequence_to_text(input_seq))

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.getcwd())
 
   #### read both positive metadata and negative metadata
  parser.add_argument('--input_pos', default='training/train-pos.txt')
  parser.add_argument('--input_neg', default='training/train-neg.txt')

  parser.add_argument('--model', default='ttsGAN')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
