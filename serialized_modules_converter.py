import torch

from itertools import zip_longest

import argparse


def saveOrderedDicts(serialized_dict, destination_dict, path):
    """
    :param serialized_dict: A dict you want to write the keys of in the first column of a csv file
    :param destination_dict: A dict you want to write the keys of in the second column of a csv file
    :param path: A string to the csv file you want to write the dicts to
    :return: None
    """
    f=open(path,"wt")
    for k1,k2 in zip_longest(serialized_dict.keys(),destination_dict.keys()):
        f.write(f"{k1},{k2}\n")
    f.close()

def csvToTranslationDict(path):
    """
    :param path: Path to the CSV file where first column is key and second is value for the dict you want to return
    :return: A dict, where keys are first column's lines and values are second column's lines from CSV at path
    """
    f=open(path)
    result={}
    for line in f:
        k,v = line.strip().split(",")
        result[k]=v
    return result

def translateStateDicts(translation_dict,origin_dict):
    """
    :param translation_dict: A dict that maps origin_dict's keys to destination key names
    :param origin_dict: A dict that contains keys you want to translate, and where the values come from
    :return: A dict where values are from origin_dict, but their keys have been translated thanks to translation_dict
    """
    return {translation_dict[k]: v for k,v in origin_dict.items()}

def exportStateDictsToCSV(path_to_serialized_dict,models_dict,save_path):
    """
    :param path_to_serialized_dict: File name of a serialized dict you want to translate the keys of
    :param models_dict: An actual dict with the keys you want to translate
    :param save_path: Where to save a csv file containing keys from above dicts
    """
    saveOrderedDicts(torch.load(path_to_serialized_dict, map_location=lambda storage,loc : storage),models_dict,save_path)
    print(f"Now, open {save_path} in the CSV editor of your choice, "
          f"and make sure that every layer name from the left column has a matching layer name in the second column")

def transferLearning(path_to_serialized_dict,model,translation_file_path):
    """
    To call this function successfully, you need a csv file containing two columns, where first maps keys from your serialized
    dict you wish to serialize, to the second that contains keys from the translated dict.

    :param path_to_serialized_dict: Path to a file containing keys you want to translate
    :param model: The model you want to transfer the data to. The keys from serilized dict will be translated to keys this model accepts.
    :param translation_file_path: Path to the CSV file containing translations (see above description)
    :return:
    """
    translation_dict = csvToTranslationDict(translation_file_path)
    serialized_dict = torch.load(path_to_serialized_dict, map_location=lambda storage,loc : storage)
    model.load_state_dict(translateStateDicts(translation_dict,serialized_dict))

def translateSerializedObjects(in_file_name,out_file_name,translation_file):
    """
    This function remaps a serialized dict (inside in_in_file_name) and writes the translation into out_out_file_name.
    This operation requires a csv translation file, where first column translates to second

    :param in_file_name: The file with a serialized dict you want to translate the keys from
    :param out_file_name: A new file where you want to write the same values as in in_file_name but with translated keys
    :param translation_file: A CSV file that translates key names from first column into other key names in the second column
    """
    translation_dict = csvToTranslationDict(translation_file)
    in_dict = torch.load(in_file_name)
    out_dict = translateStateDicts(translation_dict,in_dict)
    torch.save(out_dict,out_file_name)

parser = argparse.ArgumentParser()
parser.add_argument("input_file",help="A serialized file you want to translate the keys from")
parser.add_argument("output_file",help="A new file you want to translate the keys to")
parser.add_argument("translation_file",help="A CSV file used to map input_file's keys to output_file's ones")

args = parser.parse_args()

if __name__ == "__main__":
    translateSerializedObjects(args.input_file,args.output_file,args.translation_file)