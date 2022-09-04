import mido


def main():
    mid = mido.MidiFile("adl-piano-midi/(Da Le) Yaleo.mid", clip=True)
    for track in mid.tracks:
        print(track.name)
        for m in track:
            print(m)
    mid.save("new_song.mid")


if __name__ == "__main__":
    main()