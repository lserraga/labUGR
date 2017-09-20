"""Tool to decode audio files to WAV files."""
from __future__ import print_function
from .audioread import audio_open, NoBackendError, DecodeError
import sys
import os
import wave
import contextlib


def decode(filename):
    """
    Decodes an audio file into a wav file. Generates a new file in the 
    process.
    """
    filenamePath = os.path.abspath(os.path.expanduser(filename))
    if not os.path.exists(filenamePath):
        raise Exception ("{} not found.".format(filename))

    try:
        with audio_open(filenamePath) as f:
            with contextlib.closing(wave.open(filenamePath + '.wav', 'w')) as of:
                of.setnchannels(f.channels)
                of.setframerate(f.samplerate)
                of.setsampwidth(2)

                for buf in f:
                    of.writeframes(buf)

    except NoBackendError:
        raise Exception ("None of the avaliable backends can decode {}.".format(filename))

    except DecodeError:
        raise Exception ("{} could not be decoded.".format(filename))

if __name__ == '__main__':
    decode(sys.argv[1])