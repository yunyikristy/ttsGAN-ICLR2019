from .ttsGAN import ttsGAN
from .ttsGAN_test import ttsGAN_test

def create_model(name, hparams):
  if name == 'ttsGAN':
    return ttsGAN(hparams)
  elif name == 'ttsGAN_test':
    return ttsGAN_test(hparams)
  else:
    raise Exception('Unknown model: ' + name)
