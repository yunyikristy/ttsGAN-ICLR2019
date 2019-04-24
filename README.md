# Neural TTS Stylization with Adversarial and Collaborative Games
A tensofrlow implementation of 'Neural TTS Stylization with Adversarial and Collaborative Games', ICLR 2019

# Installing dependencies

Python >= 3.0

Tensorflow  == 1.4.0

Other libraries can be installed by running:

pip3 install -r requiredments.txt

# Test
You can use our pre-trained model to test any text you want.

You can specify any styles (emotion) by replacing the reference audio, and input any text you want by adding your text in a script '*.txt'.

We offer some examples for you test in a quick start.

1.The example texts are in 'examples_test.txt'

2.The example reference audios in './reference_audios'

3.The pre-trained model is in './pretrained-model'

To synthsizing, you need to run:

python3 test.py --model pretrained-model/model.ckpt-161000 --reference reference_audio/*.wav

To synthsizing by your own model:

python3 test.py --model 'YOUR_MODEL_PATH' --reference 'YOUR_REFERENCE_AUDIO_PATH'

# Train

	#1. Preprocess data
	
	Write your training list in this format:
	
	'mel-spec-path'|'linear-spec-path'|'lenth'|'text'|'label'
  
	Run:
	
	python3 train.py
	
	We set the default hyperparameters the same with Google's paper ' Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis'.
	
	You can adjust your own hyparameters in hparams.py
	
	Reference:
	
	Shan Yang's implementation of GST-Tacotron: https://github.com/syang1993/gst-tacotron
	
	Yuxuan Wang, Daisy Stanton, Yu Zhang, RJ Skerry-Ryan, Eric Battenberg, Joel Shor, Ying Xiao, Fei Ren, Ye Jia, Rif A. Saurous. 2018. Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis
	
	RJ Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton, Joel Shor, Ron J. Weiss, Rob Clark, Rif A. Saurous. 2018. Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron.
