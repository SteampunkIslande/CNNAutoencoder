import torch

from itertools import zip_longest


def saveOrderedDicts(serialized_dict, destination_dict, path):
    f=open(path,"wt")
    for k1,k2 in zip_longest(serialized_dict.keys(),destination_dict.keys()):
        f.write(f"{k1},{k2}\n")
    f.close()

def csvToTranslationDict(path):
    f=open(path)
    result={}
    for line in f:
        k,v = line.strip().split(",")
        result[k]=v
    return result

def translateStateDicts(translation_dict,origin_dict):
    return {translation_dict[k]: v for k,v in origin_dict.items()}