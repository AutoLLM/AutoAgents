import json
import uuid
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')

args = parser.parse_args()


def transform(conversation, search_word, notepad_word):
    for i, message in enumerate(conversation):
        conversation[i]["value"] = \
            message["value"].replace(
                "Tool_Search", search_word).replace(
                "Tool_Notepad", notepad_word)
    return conversation


input_file = args.input
output_file = args.output

with open(input_file, "r") as f:
    body = json.load(f)

result = []
for elem in body:
    search_word = str(uuid.uuid4())[:6]
    notepad_word = str(uuid.uuid4())[:6]
    elem = {
        "id": elem["id"],
        "conversations": transform(elem["conversations"], search_word, notepad_word)}
    result.append(elem)
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)
