import glob
import io
import random
def main():
    fnames = glob.glob("encoded/*")
    random.shuffle(fnames)
    f = io.open("data.txt", mode="w", encoding="ascii")
    for i, name in enumerate(fnames):
        song = open(name)
        f.write(song.read())
        f.write('\n')
        if i > 1000000:
            break

    f.close()





if __name__ == "__main__":
    main()
