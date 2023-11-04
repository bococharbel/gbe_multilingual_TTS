# gbe_multilingual_TTS

This project aims to realise a TTS system for african GBE Languages. it supports three gbe languages actually (Fongbe, Gungbe, gengbe, ...) 
and Yoruba. But we are planing to extend it to more language.

TTS system based on fastspeech2 for Gbe languages (Fongbe, Gungbe, gengbe, ...) 

## We use :

- a Fork of https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch 
(https://github.com/NVIDIA/DeepLearningExamples/)  for Fastpitch implementation  and ConvAttention (For MAS Monotonic Alignment Search with 
 Convolutional Attention to align Mel Spectrogram on input text during training )

- Conformer implementation https://github.com/sooftware/conformer (to replace Tranformers)

- SoftLengthRegulation Implementation https://github.com/LuckerYi/SoftSpeech (For duration leaning and hidden representation expanding to match mel spectrogram size)

- VQ VAE Implementation https://github.com/hhguo/MSMC-TTS (VQ feature learning)

- Fastspeech 2  implementation from  https://github.com/NVIDIA/NeMo 


## We also use scripts from :

https://github.com/alpoktem/bible2speechDB

https://github.com/neulab/AfricanVoices


Source code are partially released here.

audio samples are available at https://ttsfongbe.000webhostapp.com/index.php

# Resources

* AFRICA VOICE : https://github.com/neulab/AfricanVoices/tree/main/code/alignment
               https://github.com/neulab/AfricanVoices


* AFRICA VOICE YORUBA DATASET: https://www.africanvoices.tech/datasets


* OPEN BIBLE YORUBA : https://open.bible/resources/yoruba-davar-audio-bible/#audio


* CMU WILDERNESS : https://github.com/festvox/datasets-CMU_Wilderness


* OPEN SCRIPT : https://github.com/coqui-ai/open-bible-scripts


* BIBLE.IS : https://www.faithcomesbyhearing.com/audio-bible-resources/recordings-database


* BIBLE TTS : https://github.com/masakhane-io/bibleTTS


* MASS DATASET: https://github.com/getalp/mass-dataset


* ALFFA DATASET FONGBE : https://github.com/getalp/ALFFA_PUBLIC/tree/master/ASR/FONGBE

* TEXTE GUN ALLADA BIBLE.IS : https://www.bible.com/en-GB/bible/2405/MAT.1.BWL23

* TEXTE MINA GENGBE BIBLE.IS: https://www.bible.com/en-GB/bible/2236/MAT.1.GEN

* TEXTE FONGBE : https://www.bible.com/en-GB/bible/817/MAT.1.FON13
