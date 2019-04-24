import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio
import time

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  #print(hparams_debug_string())
  is_teacher_force = False
  reference_mel = None

  synth = Synthesizer(teacher_forcing_generating=is_teacher_force)
  synth.load(args.model, args.reference)
  base_path = get_output_base_path(args.model)

  if args.reference is not None:
    ref_wav = audio.load_wav(args.reference)
    reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
    #path = '%s_ref-%s.wav' % (base_path, os.path.splitext(os.path.basename(args.reference))[0])
    path = 'ref-%s.wav' % (os.path.splitext(os.path.basename(args.reference))[0])
  else:
      raise ValueError("You must set the reference audio.")

  
  with open('examples_test.txt', 'r') as fs:
   
      lines = fs.readlines()
      for i, line in enumerate(lines):
          args.text = line.strip().split('|')[-1]          
          
          path_id = '%d_' %(i+6)
          new_path = path_id + path
          print('Synthesizing: %s' % args.text)
          print('Output wav file: %s' % new_path)
          
          with open(new_path, 'wb') as f:
            f.write(synth.synthesize(args.text, reference_mel=reference_mel))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, help='Path to model')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--reference', default=None, help='Reference audio path')
  parser.add_argument('--text', default=None, help='Text to synthesize') 
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
