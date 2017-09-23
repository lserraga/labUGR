from .wavfile import write, read as readWav
from .decoder import decode
import os

__all__ = ['write', 'read', 'decode', 'test', 'play']

def read(filename):
	"""
    Return the sample rate (in samples/sec) and data from a audio file.
    If the audio file is not in the wav format it firsts decodes it and
    then uses read from wavfile to read it.

    Parameters
    ----------
    filename : string

    Returns
    -------
    rate : int
        Sample rate of audio file.
    data : numpy array
        Data read from audio file.
    """

	# Checking if the file exists
	if not os.path.isfile(filename):
		raise Exception ("{} not found.".format(filename))

	if filename.lower().endswith('.wav'):
		result = readWav(filename)

	# If the file is not in the wav format first decode it in a temporary
	# file and the read it 
	else:
		decode(filename)
		wav_file = "{}.wav".format(filename)
		result = readWav(wav_file)
		os.remove(wav_file)

	return result	

def play(filename):
	"""
    Plays audio files. If the audio file is not in the wav format it firsts
    decodes it and then uses read from wavfile to play it. It uses the portaudio
    library.

    Parameters
    ----------
    filename : string

    """
	import pyaudio
	import wave
	import sys

	CHUNK = 1024
	temporary = False

	# Checking if the file exists
	if not os.path.isfile(filename):
		raise Exception ("{} not found.".format(filename))

	# Decoding non-wav file
	if not filename.lower().endswith('.wav'):
		decode(filename)
		filename = "{}.wav".format(filename)
		temporary = True

	# Playing wav
	wf = wave.open(filename, 'rb')

	p = pyaudio.PyAudio()

	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	                channels=wf.getnchannels(),
	                rate=wf.getframerate(),
	                output=True)

	data = wf.readframes(CHUNK)

	while len(data) > 0:
	    stream.write(data)
	    data = wf.readframes(CHUNK)

	stream.stop_stream()
	stream.close()

	p.terminate()

	# Removing temporary file if created
	if temporary:
		os.remove(filename)


from labugr.testing.utils import PytestTester
test = PytestTester(__name__)
del PytestTester