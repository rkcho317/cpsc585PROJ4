import numpy as np
import tensorflow as tf


# There are other characters, but those are for other languages or errors, so we ignore those
alphabet = "abcdefghijklmnopqrstuvwxyz -1234567890"
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
ALPHABET_OFFSET = 3

def encode_string(inp: str, max_len: int):
    result = np.zeros(max_len + 2, np.int32)
    for index, x in enumerate(inp):
        if x not in alphabet:
            # error or foreign language
            return None
        result[index + 1] = alphabet.index(x) + ALPHABET_OFFSET
        result[0], result[len(inp) + 1] = START_TOKEN, END_TOKEN
    return result


def decode_string(inp):
    result = str()
    for x in inp:
        if x == PAD_TOKEN or x == START_TOKEN:
            continue
        if x == END_TOKEN:
            break
        result += alphabet[x - ALPHABET_OFFSET]
    return result


def load_dataset():
    data = np.load("Data/Encoded Data.npy")
    return data


if __name__ == '__main__':
    pass
