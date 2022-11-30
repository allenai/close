import json
import nltk
import openai
import random
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

openai.api_key = "YOUR_OPENAI_KEY"

harry_potter_characters = [
    "Sirius Black",
    "Cho Chang",
    "Aberforth Dumbledore",
    "Albus Dumbledore",
    "Hermione Granger",
    "Fenrir Greyback",
    "Neville Longbottom",
    "Luna Lovegood",
    "Xenophilius Lovegood",
    "Remus Lupin",
    "Draco Malfoy",
    "Lucius Malfoy",
    "Harry Potter",
    "James Potter",
    "Lily Potter",
    "Lord Voldemort",
    "Arthur Weasley",
    "Ron Weasley",
    "Severus Snape",
    "Voldemort",
    "Bellatrix Lestrange",
    "George Weasley",
    "Rubeus Hagrid",
    "Newt Scamander",
    "Delores Umbridge",
    "Minerva McGonagall",
    "Alastor Moody",
    "Molly Weasley",
    "Albus Severus Potter",
    "Gilderoy Lockhart",
    "Ginny Weasley",
    "Gellert Grindelwald",
    "Dobby"
]

def generate_coco_style_prompt(prompt):
    return """Generate caption from prompt:

person, pastry, bag:
A person putting some pastries into a bag.

statue, woman, bench, sit:
A statue of two women with purses sitting on a bench.

frisbee, catch:
A man in a grassy field about to catch a frisbee.

sculpture, living room, TV:
A living room with TV and entertainment center beside a sculpture of a torso.

girl, toddler, bed, lay:
A girl toddler laying on a day bed.

bicycle, building, door, red:
A red bicycle leaned against a red stucco building with a graffiti covered door.

sheep, grass, forest:
Sheep grazing in the grass, with a forest in the background.

cat, keyboard, yellow:
A cat sitting at a keyboard looking at a yellow post-it note with a scent written on it.

{}:""".format(prompt)

def generate_ego_centric_caption(coco_caption):
    return """Generate ego-centric caption from the given caption:

The woman is wearing and umbrella hat and on her phone.
My aunt is wearing and umbrella hat and on her phone.

A group of people sitting around two couches.
We are sitting around two couches.

A woman peaking around a pile of old refrigerators.
My girlfriend peaking around a pile of old refrigerators.

A black and white picture of a busy street, only red is in color, a red double decker bus drives down the road.
We are waiting at a busy street,  while a red double decker bus drives down the road.

A clock tower has a white pole on top.
We visited a clock tower with a white pole on top.

A road filled with cars in a desert.
We are driving on a road filled with cars in a desert.

There are horses standing in the rocks near the water.
We saw some horses standing in the rocks near the water.

A fire hydrant with grass grown around it on a curb side.
I walked past a fire hydrant with grass grown around it on a curb side.

{}
    
""".format(coco_caption)

def generate_harry_potter_prompt(prompt):
    return """Generate captions in imaginary-scenes using characters in Harry Potter, from prompt:

Harry Potter:
Harry finds an old textbook with annotations by the Half-Blood Prince, due to which he achieves success in Potions class.

Sirius Black:
Harry's life shatters before his eyes as he is bitten by Remus Lupin while Sirius Black escapes on the back of Buckbeak.

Hermione Granger:
Hermione shut the book with a snap.

Ronald Weasley:
There before him stood Ron, fully dressed but drenched to the skin, his hair plastered to his face, the sword of Gryffindor in one hand and the Horcrux dangling from its broken chain in the other.

Luna Lovegood:
Luna looked around at them, her head tilted to one side. “Yes,” she said finally.

Draco Malfoy:
Draco Malfoy smiled mirthlessly. “And you think you can stop me, Potter? You’re just a boy.”

Remus Lupin:
Remus Lupin watched as the life seeped slowly out of Ted Tonks, knowing that he could do nothing to save him.

{}:""".format(prompt)


with open('../data/coco_style_captions_train.json') as f:
    coco_data = json.load(f)
    
train_size = round(len(coco_data) * 0.7)
train_coco, val_coco = coco_data[:train_size], coco_data[train_size:]
print(f"train size = {len(train_coco)}")
print(f"val size = {len(val_coco)}")

with open('../data/ego_centric_captions_train.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)

    for i, cap in tqdm(enumerate(train_coco), leave=False):
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=generate_ego_centric_caption(cap),
            temperature=1,
            max_tokens=64,
            top_p=1,
            best_of=3,
            frequency_penalty=2,
            presence_penalty=1
        )

        for choice in response.choices:
            text = choice.text
            while text.startswith('\n') or text.startswith(' ') or text.startswith('\t'):
                text = text[1:]
            if text and text != cap:
                data.append([cap , text])
    
        # To prevent occasional OPEN AI API errors, save the data frequently.
        if i % 100 == 0:
            f.seek(0)
            json.dump(data, f, ensure_ascii=False)
    
    f.seek(0)
    json.dump(data, f, ensure_ascii=False)

with open('../data/ego_centric_captions_val.json', 'w', encoding='utf-8') as f:
    data = json.load(f)

    for i, cap in tqdm(enumerate(val_coco), leave=False):
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=generate_ego_centric_caption(cap),
            temperature=1,
            max_tokens=64,
            top_p=1,
            best_of=3,
            frequency_penalty=2,
            presence_penalty=1
        )

        for choice in response.choices:
            text = choice.text
            while text.startswith('\n') or text.startswith(' ') or text.startswith('\t'):
                text = text[1:]
            if text and text != cap:
                data.append([cap , text])
    
        # To prevent occasional OPEN AI API errors, save the data frequently.
        if i % 100 == 0:
            f.seek(0)
            json.dump(data, f, ensure_ascii=False)
    
        f.seek(0)
        json.dump(data, f, ensure_ascii=False)


with open('../data/ego_centric_captions_train.json') as f:
    data = json.load(f)
    print(len(data))

    filtered_data = []
    all_ngrams = []
    for cap in data:
        tokens = nltk.word_tokenize(cap)
        ngrams = nltk.everygrams(tokens, 4, 9)
        all_ngrams.extend(ngrams)
        filter_p = 0
        for gram in ngrams:
            if fdist[gram] >= 50:
                filter_p += fdist[gram] / 800
        r = random.uniform(0, 1)
        if r > filter_p:
            filtered_data.append(cap)

    fdist = nltk.FreqDist(all_ngrams)
    fdist_descending = OrderedDict(
        sorted(fdist.items(), key=lambda kv: kv[1], reverse=True))

with open('../data/filtered_ego_centric_captions_train.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False)
