import os
import random
import datetime
import re
import json
from itertools import permutations

base = os.path.dirname(os.path.abspath(__file__))
INPUT_INTENT_SOURCE = os.path.normpath(
    os.path.join(base, "./source/intent.txt"))

INPUT_OTHER_SOURCE = os.path.normpath(
    os.path.join(base, "./source/others.txt"))

OUTPUT = os.path.normpath(
    os.path.join(base, "./data.tsv"))


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
for line in load_file(os.path.join(base, "./source/place.txt")):
    id, name = line.split(" ")
    PLACES.append(Attribute("place", id, name))


GEOGRAPHICAL_NAMES = load_file(os.path.join(
    base, "./source/geographical_names.txt"))
STATION = load_file(os.path.join(base, "./source/station.txt"))


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
        os.path.join(base, "./source/datetime.txt"))):
    id, name = line.split(" ")
    DATES.append(Attribute("datetime", id, name))


DATE_FORMATS = ["%Y年%-m月%-d日", "%Y年%-m月", "%-m月%-d日", "%Y年", "%-m月", "%-d日"]


def get_date():
    result = []
    result.extend(DATES)
    for n in range(3):
        random_time = datetime.datetime.fromtimestamp(
            random.randint(0, 1566063664))
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
    intent, text = text.split(" ")
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

    with open(OUTPUT, "w") as out_file:
        out_file.write('\n'.join(["{}\t{}\t{}\t{}".format(
            value.intent, value.place, value.datetime, value.text) for value in result]))


if __name__ == '__main__':
    main()
