# ==============================================================================
r"""Removing unused labels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json


def _create_reduced_label_ids(category_idx_file):
    """Removes the unused labels.

    Args:
        category_idx_file: JSON file containing category_index.
    """
    counter = 0
    original_to_reduced_labels_dict = dict()
    reduced_to_original_labels_dict = dict()
    original_to_reduced_labels_file = 'original_to_reduced_labels.json'
    reduced_to_original_labels_file = 'reduced_to_original_labels.json'
    assert original_to_reduced_labels_file, "File exists : " + \
                                            original_to_reduced_labels_file
    assert reduced_to_original_labels_file, "File exists : " + \
                                            reduced_to_original_labels_file

    with open(category_idx_file, 'r') as fid:
        category_idx_original = json.load(fid)
    for k in category_idx_original.keys():
        counter += 1
        original_to_reduced_labels_dict[k] = counter
        reduced_to_original_labels_dict[counter] = k

    with open(original_to_reduced_labels_file, 'w') as fid:
        json.dump(original_to_reduced_labels_dict, fid)
    with open(reduced_to_original_labels_file, 'w') as fid:
        json.dump(reduced_to_original_labels_dict, fid)

    print("Created mappings between original and reduced training labels...")


def _create_rearranged_label_ids(category_idx_file):
    """Rearranges the labels.

    Args:
        category_idx_file: JSON file containing category_index.
    """
    counter = 0
    original_to_rearranged_labels_dict = dict()
    rearranged_to_original_labels_dict = dict()
    original_to_rearranged_labels_file = 'original_to_rearranged_labels.json'
    rearranged_to_original_labels_file = 'rearranged_to_original_labels.json'
    assert original_to_rearranged_labels_file,\
        "File exists : " + original_to_rearranged_labels_file
    assert rearranged_to_original_labels_file,\
        "File exists : " + rearranged_to_original_labels_file

    with open(category_idx_file, 'r') as fid:
        category_idx_original = json.load(fid)
    thing_list = []
    stuff_list = []
    for key, value in category_idx_original.items():
        if value['isthing'] == 0:
            stuff_list.append(key)
        else:
            thing_list.append(key)
    total_list = thing_list + stuff_list
    for i in total_list:
        counter += 1
        original_to_rearranged_labels_dict[int(i)] = int(counter)
        rearranged_to_original_labels_dict[int(counter)] = int(i)

    with open(original_to_rearranged_labels_file, 'w') as fid:
        json.dump(original_to_rearranged_labels_dict, fid)
    with open(rearranged_to_original_labels_file, 'w') as fid:
        json.dump(rearranged_to_original_labels_dict, fid)

    print("Created mappings between original and rearranged training labels...")


def main():
    # category_idx_file = 'category_index_coco.json'
    # assert category_idx_file, "File doesnt exist : " + category_idx_file
    # _create_reduced_label_ids(category_idx_file)
    category_idx_file = 'category_index_mapillary.json'
    assert category_idx_file, "File doesnt exist : " + category_idx_file
    _create_rearranged_label_ids(category_idx_file)


if __name__ == '__main__':
    main()
