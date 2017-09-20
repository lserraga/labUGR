from .wavfile import write, read as readWav
from .decoder import decode

__all__ = ['write', 'read', 'decode', 'test']

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
	
	import os

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


from labugr.testing.utils import PytestTester
test = PytestTester(__name__)
del PytestTester