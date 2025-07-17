import collections
import json
import os
import re
import collections.abc


def update_item(item, old, new):
    """
    :param item: can be a string, list, or anything else
    :param old: string to be replaced
    :param new: replacement
    :return: item after string replacement
    """

    if isinstance(item, str):
        return re.sub(old, new, item)

    # For lists, iterative over the items
    elif isinstance(item, list):
        # Start with an empty list
        updated_list = []
        for i in range(len(item)):
            # Take list item i, recursively update
            updated_list.append(update_item(item[i], old, new))

        return updated_list

    else:
        # For other data types, do nothing
        return item


def restore_args(model_dir, overwrite_path=None):
    args_file = os.path.join(model_dir, "args.json")
    with open(args_file, "r") as f:
        restored_args = json.load(f)

    # Overwrite paths if necessary
    if overwrite_path:
        # overwrite_path can be a dictionary
        if isinstance(overwrite_path, collections.abc.Mapping):
            for old, new in overwrite_path.items():
                for k, v in restored_args.items():
                    restored_args[k] = update_item(v, old, new)

        else:
            # or a simple tuple
            old, new = overwrite_path
            for k, v in restored_args.items():
                restored_args[k] = update_item(v, old, new)

    restored_args["model_checkpoint"] = os.path.join(model_dir, "model", "model.pt")
    return restored_args


def is_list_or_tuple(obj):
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)
