from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
import glob
import os
MAX_FILES = 100000
import random
def main():
    fnames = glob.glob("Outmidis/*")
    for name in fnames:
        mido_obj =mid_parser.MidiFile()
        beat_resol = mido_obj.ticks_per_beat
        track = ct.Instrument(program=0, is_drum=False, name='example track')
        mido_obj.instruments = [track]
        f = open(name)
        notes = f.read().split()
        start = 0
        fname = name.split("/")[1][:-4]
        for hash in notes:
            if not ":" in hash: continue
            prev, pitch = map(int, hash.split(":"))
            start += prev * beat_resol//12
            end = start + beat_resol
            note = ct.Note(start=start, end=end, pitch=pitch, velocity=63)
            mido_obj.instruments[0].notes.append(note)
        mido_obj.dump("TransformerMidi/" + fname + ".mid")

if __name__ == "__main__":
    main()
