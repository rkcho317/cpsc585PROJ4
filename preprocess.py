import csv
import numpy as np
import tensorflow as tf
from utils import *


def main():
    with open('Data/simpsons_script.csv', newline='', encoding='utf-8') as csvfile:
        data_set = csv.reader(csvfile, delimiter=',')
        normalized_text = [x[3] for i, x in enumerate(data_set) if i > 0]

    max_len = 0
    for x in normalized_text:
        max_len = max(len(x), max_len)

    encoded_text = []
    for x in normalized_text:
        temp = encode_string(x, max_len)
        if temp is not None:
            encoded_text.append(temp)
    encoded_text = np.array(encoded_text)
    np.save("Data/Encoded Data", encoded_text)


if __name__ == '__main__':
    main()
