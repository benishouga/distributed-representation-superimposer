import os
import json

base = os.path.dirname(os.path.abspath(__file__))
INPUT_OTHER_SOURCE = os.path.normpath(os.path.join(
    base, "./source/projectnextnlp-chat-dialogue-corpus/json/rest1046"))

def load_others():
    files = os.listdir(INPUT_OTHER_SOURCE)
    result = []
    for filename in files:
        path = os.path.join(INPUT_OTHER_SOURCE, filename)
        file = open(path, "r")
        copus_json = json.load(file)
        for turn in copus_json["turns"]:
            if turn["speaker"] == "U":
                result.append(turn["utterance"])

    return result

def main():
    others = load_others()
    for other in others:
        print(other)

if __name__ == '__main__':
    main()
