from glob import glob
import os
from utils.args import is_list_or_tuple


FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".webp")


def find_files(directory, file_extensions=FILE_EXTENSIONS):
    """
    Find files in a given directory. Does not search subdirectories.
    :param directory: input directory
    :param file_extensions: tuple of file extensions
    :return: list of filepaths
    """

    # File extensions must be a tuple
    if not isinstance(file_extensions, tuple):
        if isinstance(file_extensions, list):
            # Auto-convert lists to tuples
            file_extensions = tuple(file_extensions)
        else:
            raise ValueError("Candidate file extensions must be a tuple")

    # Do not search subdirectories
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filepath.endswith(file_extensions):
            files.append(filepath)

    return files

def find_files_recursively(directory, file_extensions=FILE_EXTENSIONS):
    """
    Recursively find all the files with the given file extensions
    :param directory: input directory to search recursively
    :param file_extensions: list of file extensions
    :return: iterator that yields absolute paths
    """
    if not is_list_or_tuple(file_extensions):
        raise ValueError("Expected file extensions to be a list or tuple")

    file_extensions = set(file_extensions)
    for root, dirs, files in os.walk(directory):
        for basename in files:
            ext = os.path.splitext(basename)[1].lower()
            if ext in file_extensions:
                filepath = os.path.join(root, basename)
                yield filepath


def glob_files_recursively(directory, file_extensions=FILE_EXTENSIONS):
    """
    Find all the files with the given file extension.

    Slower than the os.walk method.

    :param directory: input directory to search recursively
    :param file_extensions: list of file extensions
    :return: list of files
    """
    if not is_list_or_tuple(file_extensions):
        raise ValueError("Expected file extensions to be a list or tuple")

    file_extensions = set(file_extensions)
    filepaths = []
    for file_type in file_extensions:
        filepaths.extend(glob(os.path.join(directory, "**", "*" + file_type), recursive=True))
    return filepaths


def resolve_path(filepath):
    """
    Resolves wildcards in a given filepath. Verifies that the file existence.
    :param filepath: filepath, can contain wildcards
    :return: filepath
    """
    # Check if the filepath contains a wildcard
    if "*" in filepath:
        # Yes, then it is a glob pattern
        matches = glob(filepath)
        if len(matches) == 1:
            filepath = matches[0]
        else:
            raise ValueError(f"Expected to find exactly one file mathing \"{filepath}\", but found {len(matches)}.")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Given file \"{filepath}\" does not exist.")

    return filepath
