import json
import numpy as np
import re

# def add_pad_to_vocab(vocab):
#     new_vocab = {"PAD": 0}
#     for k in vocab:
#         new_vocab[k] = vocab[k] + 1
#     return new_vocab
#
#
# def combine_vocabs(vocabs, vocab_types):
#     new_vocab = {}
#     for type, vocab in zip(vocab_types, vocabs):
#         for k in vocab:
#             new_vocab["{}:{}".format(type, k)] = len(new_vocab)
#     return new_vocab
#
#
# def add_token_to_vocab(vocab):
#     new_vocab = {"MASK": 0}
#     for k in vocab:
#         new_vocab[k] = vocab[k] + 1
#     return new_vocab
#
#
# def tokenize_circle_specification(circle_specification):
#     tokenized = {}
#     # min 0, max 0.5, increment 0.05, 10 discrete values
#     tokenized["radius"] = int(circle_specification["radius"] / 0.05)
#
#     # min 0, max 1, increment 0.10, 10 discrete values
#     tokenized["position_x"] = int(circle_specification["position"][0] / 0.10)
#
#     # min -0.5, max 0.5, increment 0.10, 10 discrete values
#     tokenized["position_y"] = int(circle_specification["position"][1] / 0.10)
#
#     # min -3.14, max 3.14, increment 3.14 / 18, 36 discrete values
#     tokenized["rotation"] = int((circle_specification["rotation"][2] + 3.14) / (3.14 / 18))
#
#     uniform_angle_vocab = {"False": 0, "True": 1}
#     tokenized["uniform_angle"] = uniform_angle_vocab[circle_specification["uniform_angle"]]
#
#     face_center_vocab = {"False": 0, "True": 1}
#     tokenized["face_center"] = face_center_vocab[circle_specification["face_center"]]
#
#     angle_ratio_vocab = {0.5: 0, 1.0: 1}
#     tokenized["angle_ratio"] = angle_ratio_vocab[circle_specification["angle_ratio"]]
#
#     # heights min 0.0, max 0.5
#     # volumn min 0.0, max 0.012
#
#     return tokenized
#
#
# def build_vocab(old_vocab_file, new_vocab_file):
#     with open(old_vocab_file, "r") as fh:
#         vocab_json = json.load(fh)
#
#     vocabs = {}
#     vocabs["class"] = vocab_json["class_to_idx"]
#     vocabs["size"] = vocab_json["size_to_idx"]
#     vocabs["color"] = vocab_json["color_to_idx"]
#     vocabs["material"] = vocab_json["material_to_idx"]
#     vocabs["comparator"] = {"less": 1, "greater": 2, "equal": 3}
#
#     vocabs["radius"] = (0.0, 0.5, 10)
#     vocabs["position_x"] = (0.0, 1.0, 10)
#     vocabs["position_y"] = (-0.5, 0.5, 10)
#     vocabs["rotation"] = (-3.14, 3.14, 36)
#     vocabs["height"] = (0.0, 0.5, 10)
#     vocabs["volumn"] = (0.0, 0.012, 10)
#
#     vocabs["uniform_angle"] = {"False": 0, "True": 1}
#     vocabs["face_center"] = {"False": 0, "True": 1}
#     vocabs["angle_ratio"] = {0.5: 0, 1.0: 1}
#
#     with open(new_vocab_file, "w") as fh:
#         json.dump(vocabs, fh)


class Tokenizer:
    """
    We want to build a tokenizer that tokenize words, features, and numbers.

    This tokenizer should also allow us to sample random values.

    For discrete values, we store mapping from the value to an id
    For continuous values, we store min, max, and number of bins after discretization

    """

    def __init__(self, vocab_file):

        self.vocab_file = vocab_file
        with open(self.vocab_file, "r") as fh:
            self.type_vocabs = json.load(fh)

        self.vocab = {"PAD": 0, "CLS": 1}
        self.discrete_types = set()
        self.continuous_types = set()
        self.build_one_vocab()

        self.object_position_vocabs = {}
        self.build_object_position_vocabs()

    def build_one_vocab(self):
        print("\nBuild one vacab for everything...")

        for typ, vocab in self.type_vocabs.items():
            if typ == "comparator":
                continue

            if typ in ["obj_x", "obj_y", "obj_z", "obj_rr", "obj_rp", "obj_ry",
                       "struct_x", "struct_y", "struct_z", "struct_rr", "struct_rp", "struct_ry"]:
                continue

            if type(vocab) == dict:
                self.vocab["{}:{}".format(typ, "MASK")] = len(self.vocab)

                for v in vocab:
                    assert ":" not in v
                    self.vocab["{}:{}".format(typ, v)] = len(self.vocab)
                self.discrete_types.add(typ)

            elif type(vocab) == tuple or type(vocab) == list:
                self.vocab["{}:{}".format(typ, "MASK")] = len(self.vocab)

                for c in self.type_vocabs["comparator"]:
                    self.vocab["{}:{}".format(typ, c)] = len(self.vocab)

                min_value, max_value, num_bins = vocab
                for i in range(num_bins):
                    self.vocab["{}:{}".format(typ, i)] = len(self.vocab)
                self.continuous_types.add(typ)
            else:
                raise TypeError("The dtype of the vocab cannot be handled: {}".format(vocab))

        print("The vocab has {} tokens: {}".format(len(self.vocab), self.vocab))

    def build_object_position_vocabs(self):
        print("\nBuild vocabs for object position")
        for typ in ["obj_x", "obj_y", "obj_z", "obj_rr", "obj_rp", "obj_ry",
                    "struct_x", "struct_y", "struct_z", "struct_rr", "struct_rp", "struct_ry"]:
            self.object_position_vocabs[typ] = {"PAD": 0, "MASK": 1}

            if typ not in self.type_vocabs:
                continue
            min_value, max_value, num_bins = self.type_vocabs[typ]
            for i in range(num_bins):
                self.object_position_vocabs[typ]["{}".format(i)] = len(self.object_position_vocabs[typ])
            print("The {} vocab has {} tokens: {}".format(typ, len(self.object_position_vocabs[typ]), self.object_position_vocabs[typ]))

    def get_object_position_vocab_sizes(self):
        return len(self.object_position_vocabs["position_x"]), len(self.object_position_vocabs["position_y"]), len(self.object_position_vocabs["rotation"])

    def get_vocab_size(self):
        return len(self.vocab)

    def tokenize_object_position(self, value, typ):
        assert typ in ["obj_x", "obj_y", "obj_z", "obj_rr", "obj_rp", "obj_ry",
                       "struct_x", "struct_y", "struct_z", "struct_rr", "struct_rp", "struct_ry"]
        if value == "MASK" or value == "PAD":
            return self.object_position_vocabs[typ][value]
        elif value == "IGNORE":
            # Important: used to avoid computing loss. -100 is the default ignore_index for NLLLoss
            return -100
        else:
            min_value, max_value, num_bins = self.type_vocabs[typ]
            assert min_value <= value <= max_value, value
            dv = min(int((value - min_value) / ((max_value - min_value) / num_bins)), num_bins - 1)
            return self.object_position_vocabs[typ]["{}".format(dv)]

    def tokenize(self, value, typ=None):
        if value in ["PAD", "CLS"]:
            idx = self.vocab[value]
        else:
            if typ is None:
                raise KeyError("Type cannot be None")

            if typ[-2:] == "_c" or typ[-2:] == "_d":
                typ = typ[:-2]

            if typ in self.discrete_types:
                idx = self.vocab["{}:{}".format(typ, value)]
            elif typ in self.continuous_types:
                if value == "MASK" or value in self.type_vocabs["comparator"]:
                    idx = self.vocab["{}:{}".format(typ, "MASK")]
                else:
                    min_value, max_value, num_bins = self.type_vocabs[typ]
                    assert min_value <= value <= max_value, "type {} value {} exceeds {} and {}".format(typ, value, min_value, max_value)
                    dv = min(int((value - min_value) / ((max_value - min_value) / num_bins)), num_bins - 1)
                    # print(value, dv, "{}:{}".format(typ, dv))
                    idx = self.vocab["{}:{}".format(typ, dv)]
            else:
                raise KeyError("Do not recognize the type {} of the given token: {}".format(typ, value))
        return idx

    def get_valid_random_value(self, typ):
        """
        Get a random value for the given typ
        :param typ:
        :return:
        """
        if typ[-2:] == "_c" or typ[-2:] == "_d":
            typ = typ[-2:]

        candidate_values = []
        for v in self.vocab:
            if v in ["PAD", "CLS"]:
                continue
            ft, fv = v.split(":")
            if typ == ft and fv != "MASK" and fv not in self.type_vocabs["comparator"]:
                candidate_values.append(v)
        assert len(candidate_values) != 0
        typed_v = np.random.choice(candidate_values)
        value = typed_v.split(":")[1]

        if typ in self.discrete_types:
            return value
        elif typ in self.continuous_types:
            min_value, max_value, num_bins = self.type_vocabs[typ]
            return min_value + ((max_value - min_value) / num_bins) * int(value)
        else:
            raise KeyError("Do not recognize the type {} of the given token".format(typ))

    def get_all_values_of_type(self, typ):
        """
        Get all values for the given typ
        :param typ:
        :return:
        """
        if typ[-2:] == "_c" or typ[-2:] == "_d":
            typ = typ[-2:]

        candidate_values = []
        for v in self.vocab:
            if v in ["PAD", "CLS"]:
                continue
            ft, fv = v.split(":")
            if typ == ft and fv != "MASK" and fv not in self.type_vocabs["comparator"]:
                candidate_values.append(v)
        assert len(candidate_values) != 0
        values = [typed_v.split(":")[1] for typed_v in candidate_values]

        if typ in self.discrete_types:
            return values
        else:
            raise KeyError("Do not recognize the type {} of the given token".format(typ))

    def convert_to_natural_sentence(self, template_sentence):

        # select objects that are [red, metal]
        # select objects that are [larger, taller] than the [], [], [] object
        # select objects that have the same [color, material] of the [], [], [] object

        natural_sentence_templates = ["select objects that are {}.",
                                      "select objects that have {} {} {} the {}.",
                                      "select objects that have the same {} as the {}."]

        v, t = template_sentence[0]
        if t[-2:] == "_c" or t[-2:] == "_d":
            t = t[:-2]

        if v != "MASK" and t in self.discrete_types:
            natural_sentence_template = natural_sentence_templates[0]
            if t == "class":
                natural_sentence = natural_sentence_template.format(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', v)[0].lower())
            else:
                natural_sentence = natural_sentence_template.format(v)
        else:
            anchor_obj_properties = []
            class_reference = None
            for token in template_sentence[1:]:
                if token[0] != "PAD":
                    if token[1] == "class":
                        class_reference = token[0]
                    else:
                        anchor_obj_properties.append(token[0])
            # order the properties
            anchor_obj_des = ", ".join(anchor_obj_properties)
            if class_reference is None:
                anchor_obj_des += " object"
            else:
                anchor_obj_des += " {}".format(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', class_reference)[0].lower())

            if v == "MASK":
                natural_sentence_template = natural_sentence_templates[2]
                anchor_type = t
                natural_sentence = natural_sentence_template.format(anchor_type, anchor_obj_des)
            elif t in self.continuous_types:
                natural_sentence_template = natural_sentence_templates[1]
                if v == "equal":
                    jun = "as"
                else:
                    jun = "than"
                natural_sentence = natural_sentence_template.format(v, t, jun, anchor_obj_des)
            else:
                raise NotImplementedError

        return natural_sentence

    def prepare_grounding_reference(self):
        goal = {"rearrange": {"features": []},
                "anchor": {"features": []}}
        discrete_type = ["class", "material", "color"]
        continuous_type = ["volumn", "height"]

        print("#"*50)
        print("Preparing referring expression")

        refer_type = verify_input("direct (1) or relational reference (2)? ", [1, 2], int)
        if refer_type == 1:

            # 1. no anchor
            t = verify_input("desired type: ", discrete_type, None)
            v = verify_input("desired value: ", self.get_all_values_of_type(t), None)

            goal["rearrange"]["features"].append({"comparator": None, "type": t, "value": v})

        elif refer_type == 2:

            value_type = verify_input("discrete (1) or continuous relational reference (2)? ", [1, 2], int)
            if value_type == 1:
                t = verify_input("desired type: ", discrete_type, None)
                # 2. discrete
                goal["rearrange"]["features"].append({"comparator": None, "type": t, "value": None})
            elif value_type == 2:
                comp = verify_input("desired comparator: ", list(self.type_vocabs["comparator"].keys()), None)
                t = verify_input("desired type: ", continuous_type, None)
                # 3. continuous
                goal["rearrange"]["features"].append({"comparator": comp, "type": t, "value": None})

            num_f = verify_input("desired number of features for the anchor object: ", [1, 2, 3], int)
            for i in range(num_f):
                t = verify_input("desired type: ", discrete_type, None)
                v = verify_input("desired value: ", self.get_all_values_of_type(t), None)
                goal["anchor"]["features"].append({"comparator": None, "type": t, "value": v})

        return goal

    def convert_structure_params_to_natural_language(self, sentence):

        # ('circle', 'shape'), (-1.3430555575431449, 'rotation'), (0.3272675147405848, 'position_x'), (-0.03104362197706456, 'position_y'), (0.04674859577847633, 'radius')

        shape = None
        x = None
        y = None
        rot = None
        size = None

        for param in sentence:
            if param[0] == "PAD":
                continue

            v, t = param
            if t == "shape":
                shape = v
            elif t == "position_x":
                dv = self.discretize(v, t)
                if dv == 0:
                    x = "bottom"
                elif dv == 1:
                    x = "middle"
                elif dv == 2:
                    x = "top"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))
            elif t == "position_y":
                dv = self.discretize(v, t)
                if dv == 0:
                    y = "right"
                elif dv == 1:
                    y = "center"
                elif dv == 2:
                    y = "left"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))
            elif t == "radius":
                dv = self.discretize(v, t)
                if dv == 0:
                    size = "small"
                elif dv == 1:
                    size = "medium"
                elif dv == 2:
                    size = "large"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))
            elif t == "rotation":
                dv = self.discretize(v, t)
                if dv == 0:
                    rot = "north"
                elif dv == 1:
                    rot = "east"
                elif dv == 2:
                    rot = "south"
                elif dv == 3:
                    rot = "west"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))

        natural_sentence = "" # "{} {} in the {} {} of the table facing {}".format(size, shape, x, y, rot)

        if size:
            natural_sentence += "{}".format(size)
        if shape:
            natural_sentence += " {}".format(shape)
        if x:
            natural_sentence += " in the {}".format(x)
        if y:
            natural_sentence += " {} of the table".format(y)
        if rot:
            natural_sentence += " facing {}".format(rot)

        natural_sentence = natural_sentence.strip()

        return natural_sentence

    def convert_structure_params_to_type_value_tuple(self, sentence):

        # ('circle', 'shape'), (-1.3430555575431449, 'rotation'), (0.3272675147405848, 'position_x'), (-0.03104362197706456, 'position_y'), (0.04674859577847633, 'radius')

        shape = None
        x = None
        y = None
        rot = None
        size = None

        for param in sentence:
            if param[0] == "PAD":
                continue

            v, t = param
            if t == "shape":
                shape = v
            elif t == "position_x":
                dv = self.discretize(v, t)
                if dv == 0:
                    x = "bottom"
                elif dv == 1:
                    x = "middle"
                elif dv == 2:
                    x = "top"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))
            elif t == "position_y":
                dv = self.discretize(v, t)
                if dv == 0:
                    y = "right"
                elif dv == 1:
                    y = "center"
                elif dv == 2:
                    y = "left"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))
            elif t == "radius":
                dv = self.discretize(v, t)
                if dv == 0:
                    size = "small"
                elif dv == 1:
                    size = "medium"
                elif dv == 2:
                    size = "large"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))
            elif t == "rotation":
                dv = self.discretize(v, t)
                if dv == 0:
                    rot = "north"
                elif dv == 1:
                    rot = "east"
                elif dv == 2:
                    rot = "south"
                elif dv == 3:
                    rot = "west"
                else:
                    raise KeyError("key {} not found in {}".format(v, self.type_vocabs[t]))

        # rotation, shape, size, x, y
        type_value_tuple_init = [("rotation", rot), ("shape", shape), ("size", size), ("x", x), ("y", y)]
        type_value_tuple = []
        for type_value in type_value_tuple_init:
            if type_value[1] is not None:
                type_value_tuple.append(type_value)

        type_value_tuple = tuple(sorted(type_value_tuple))
        return type_value_tuple

    def discretize(self, v, t):
        min_value, max_value, num_bins = self.type_vocabs[t]
        assert min_value <= v <= max_value, "type {} value {} exceeds {} and {}".format(t, v, min_value, max_value)
        dv = min(int((v - min_value) / ((max_value - min_value) / num_bins)), num_bins - 1)
        return dv


class ContinuousTokenizer:
    """
    This tokenizer is for testing not discretizing structure parameters
    """

    def __init__(self):

        print("WARNING: Current continous tokenizer does not support multiple shapes")

        self.continuous_types = ["rotation", "position_x", "position_y", "radius"]
        self.discrete_types = ["shape"]

    def tokenize(self, value, typ=None):
        if value == "PAD":
            idx = 0.0
        else:
            if typ is None:
                raise KeyError("Type cannot be None")
            elif typ in self.discrete_types:
                idx = 1.0
            elif typ in self.continuous_types:
                idx = value
            else:
                raise KeyError("Do not recognize the type {} of the given token: {}".format(typ, value))
        return idx


if __name__ == "__main__":
    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")
    # print(tokenizer.get_all_values_of_type("class"))
    # print(tokenizer.get_all_values_of_type("color"))
    # print(tokenizer.get_all_values_of_type("material"))
    #
    # for type in tokenizer.type_vocabs:
    #     print(type, tokenizer.type_vocabs[type])

    tokenizer.prepare_grounding_reference()

    # for i in range(100):
    #     types = list(tokenizer.continuous_types) + list(tokenizer.discrete_types)
    #     for t in types:
    #         v = tokenizer.get_valid_random_value(t)
    #         print(v)
    #         print(tokenizer.tokenize(v, t))

    # build_vocab("/home/weiyu/data_drive/examples_v4/leonardo/vocab.json", "/home/weiyu/data_drive/examples_v4/leonardo/type_vocabs.json")