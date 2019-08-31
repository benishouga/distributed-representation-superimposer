import os
import time
import random
import datetime
import re
import json
from itertools import permutations
from extractor import Extractor

base = os.path.dirname(os.path.abspath(__file__))
INPUT_INTENT_SOURCE = os.path.normpath(
    os.path.join(base, "./data/source/intent.txt"))

INPUT_OTHER_SOURCE = os.path.normpath(
    os.path.join(base, "./data/source/others.txt"))

OUTPUT_CATALOGER = os.path.normpath(
    os.path.join(base, "./data/data_cataloger.tsv"))

OUTPUT_SUPERIMPOSER = os.path.normpath(
    os.path.join(base, "./data/data_superimposer.tsv"))


REGEX_FIND_HOLDER = re.compile(r'\{(.*?)\}')


class Intent:
    def __init__(self, text, intent, datetime=None, place=None):
        self.text = text
        self.intent = intent
        self.place = place
        self.datetime = datetime


class Attribute:
    def __init__(self, type, id, name):
        self.type = type
        self.id = id
        self.name = name


class Pack:
    def __init__(self, text, attributes):
        self.text = text
        self.attributes = attributes


def load_file(file_path):
    result = []
    with open(file_path, "r") as lines:
        for line in lines:
            result.append(line[:-1])
    return result


PLACES = []
for line in load_file(os.path.join(base, "./data/source/place.txt")):
    id, name = line.split("\t")
    PLACES.append(Attribute("place", id, name))


GEOGRAPHICAL_NAMES = load_file(os.path.join(
    base, "./data/source/geographical_names.txt"))
STATION = load_file(os.path.join(base, "./data/source/station.txt"))


def get_place():
    result = []
    result.extend(PLACES)
    for n in range(3):
        result.append(Attribute("place", "geographical_names",
                                random.choice(GEOGRAPHICAL_NAMES)))
    for n in range(3):
        result.append(Attribute("place", "station", random.choice(STATION)))
    return result


DATES = []
for line in load_file(os.path.normpath(
        os.path.join(base, "./data/source/datetime.txt"))):
    id, name = line.split("\t")
    DATES.append(Attribute("datetime", id, name))


DATE_FORMATS = ["%-m月%-d日", "%-m月", "%-d日"]


def get_date():
    result = []
    result.extend(DATES)
    for n in range(2):
        random_time = datetime.datetime.fromtimestamp(
            random.randint(1565000000, 1700000000))
        result.append(
            Attribute("datetime", "absolute", random_time.strftime(random.choice(DATE_FORMATS))))
    return result


def extract_type(type):
    if type == "place":
        return get_place()
    elif type == "date":
        return get_date()
    return []


def make_attributes_comb(array_in_array):
    current = []
    for next in array_in_array:
        if len(current) == 0:
            for n in next:
                current.append([n])
            continue
        new_array = []
        for x in current:
            for y in next:
                v = []
                v.extend(x)
                v.append(y)
                new_array.append(v)
        current = new_array
    return current


def extract(holder):
        # {date|place+の}
        # {date|place+は}
        # {place}
        # {date}

    splited = holder.split("+")
    types_text = splited[0]
    particle = ""
    if len(splited) == 2:
        particle = splited[1]

    types = types_text.split("|")
    combo = []

    for num in range(len(types)):
        combo.extend(permutations(types, num + 1))

    result = []
    for permutation in combo:
        extracted_types = [extract_type(type) for type in permutation]
        attributes_comb = make_attributes_comb(extracted_types)
        for attributes in attributes_comb:
            text = "の".join([attr.name for attr in attributes]) + particle
            result.append(Pack(text, attributes))
    return result


def build(text):
    intent, text = text.split("\t")
    reg_result = REGEX_FIND_HOLDER.search(text)
    if reg_result == None:
        return text
    prefix = text[0:reg_result.start()]
    suffix = text[reg_result.end():]
    holder = reg_result.group(1)
    extracted_values = extract(holder)
    result = []
    for value in extracted_values:
        datetime = None
        place = None
        for attr in value.attributes:
            if attr.type == "place":
                place = attr.id
            elif attr.type == "datetime":
                datetime = attr.id
        result.append(Intent(prefix + value.text +
                             suffix, intent, datetime, place))
    return result


def main():
    result = []

    intents = load_file(INPUT_INTENT_SOURCE)
    for line in intents:
        result.extend(build(line))

    others = load_file(INPUT_OTHER_SOURCE)
    for other in others:
        result.append(Intent(other, "other"))

    extractor = Extractor('../Japanese_L-12_H-768_A-12_E-30_BPE/')

    # # カウント数分だけ
    # count = 32
    # i = 0
    # start_time = time.time()
    # print("start")
    # for value in result:
    #     if i < count:
    #         value.dr = [format(v, 'e') for v in extractor.extract(value.text)]
    #         i = i + 1
    #     else:
    #         value.dr = None
    # finish = time.time()
    # duration = finish - start_time
    # print("duration on one text", duration / count)
    # print("expected duration", duration * 38768 / count / 60, "min")

    # # 全部やらない
    # for value in result:
    #     value.dr = None

    # 全部やる
    for value in result:
        value.dr = [format(v, 'e') for v in extractor.extract(value.text)]

    with open(OUTPUT_CATALOGER, "w") as out_file:
        contents = []
        for value in result:
            contents.append("\t".join([
                value.intent,
                value.place or "",
                value.datetime or "",
                value.text,
                ",".join(value.dr) if value.dr else ""
            ]))
        out_file.write('\n'.join(contents))

    with open(OUTPUT_SUPERIMPOSER, "w") as out_file:
        contents = []
        for one in result:
            two = random.choice(result)
            expect = {
                "intent": one.intent if two.intent is None else two.intent,
                "place": one.place if two.place is None else two.place,
                "datetime": one.datetime if two.datetime is None else two.datetime
            }
            contents.append("\t".join([
                expect["intent"],
                expect["place"] or "",
                expect["datetime"] or "",
                one.text,
                ",".join(one.dr) if one.dr else "",
                two.text,
                ",".join(two.dr) if two.dr else ""
            ]))
        out_file.write('\n'.join(contents))

if __name__ == '__main__':
    main()
