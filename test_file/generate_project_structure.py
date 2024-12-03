import os
from metro_data_convertor.Find_project_root import Find_project_root

def generate_project_structure(root_dir, output_file, ignore_dirs):
    with open(output_file, 'w') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if any(ignored in dirpath for ignored in ignore_dirs):
                continue
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f'{indent}{os.path.basename(dirpath)}/\n')
            sub_indent = ' ' * 4 * (level + 1)
            for filename in filenames:
                f.write(f'{sub_indent}{filename}\n')


def int_to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num


def increment_letter(letter_seq, is_upper=False):
    """
    The letter sequence increments, such as a -> b -> z -> aa -> ab or A -> B -> Z -> AA -> AB.
    letter_seq: The current letter sequence.
    is_upper: Indicates if the sequence is uppercase; True for uppercase, False for lowercase.
    return: The incremented letter sequence.
    """
    if not letter_seq:
        return 'A' if is_upper else 'a'

    last_char = letter_seq[-1]
    if is_upper:
        if last_char == 'Z':
            return increment_letter(letter_seq[:-1], is_upper=True) + 'A' if letter_seq[:-1] else 'AA'
        else:
            return letter_seq[:-1] + chr(ord(last_char) + 1)
    else:
        if last_char == 'z':
            return increment_letter(letter_seq[:-1], is_upper=False) + 'a' if letter_seq[:-1] else 'aa'
        else:
            return letter_seq[:-1] + chr(ord(last_char) + 1)


def increment_greek_letter(letter_seq):
    """
    The Greek letter sequence increments, such as α -> β -> γ, and repeats when it exceeds the range.
    letter_seq: The current Greek letter sequence.
    return: The incremented Greek letter sequence.
    """
    greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ',
                     'φ', 'χ', 'ψ', 'ω']
    current_index = greek_letters.index(letter_seq) if letter_seq in greek_letters else -1
    next_index = (current_index + 1) % len(greek_letters)  # 超出范围时重复使用
    return greek_letters[next_index]


def should_print_hiam_directory(dirpath, filenames):
    """
    Determine whether a folder that starts with HIAM should be printed.
    dirpath: The folder path.
    filenames: A list of filenames in the folder.
    return: Return True if the folder should be printed; otherwise, return False.
    """
    if os.path.basename(dirpath).startswith("HIAM"):
        # 如果该文件夹只包含一个文件，且文件名为 info.log，则不打印
        if len(filenames) == 1 and filenames[0] == 'info.log':
            return False
    return True


def generate_two_formats_project_structure(root_dir, output_file, ignore_dirs):
    """
    Generate the directory structure of a project and write it to a file, ignoring specified files in the given directories but preserving the directory and subdirectory structure:

    Level 1 directories: Use a counting format of 1. 2. 3., incrementing each time a level 1 directory appears.
    Level 2 directories: Use a counting format of 1) 2) 3., incrementing each time a level 2 directory appears, and resetting when the level 1 directory changes.
    Level 3 directories: Use a counting format of (1) (2) (3), incrementing each time a level 3 directory appears, and resetting when the level 2 directory changes.
    .py files: Use a counting format of [1] [2].
    .yaml files: Use a counting format of a. b. c. -> z. aa. ab.
    .pkl, .pt, .xlsx files: Use a counting format of A. B. C. -> Z. AA. AB.
    Other unmentioned file types: Use Greek letters α. β. γ. as the counting format.
    If a folder starts with "HIAM" and contains only one file, info.log, it should not be printed. All __pycache__ folders and their subdirectories should not be printed.
    root_dir: The root directory of the project.
    output_file: The name of the output file.
    ignore_dirs: A list of directories from which to ignore files, but these directories and their subdirectories will still be displayed.
    """
    with open(output_file, 'w') as f:
        f.write("Original directory structure:\n")
        printed_hiam = False
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if '__pycache__' in dirpath:
                continue

            if not should_print_hiam_directory(dirpath, filenames):
                continue

            if os.path.basename(dirpath).startswith("HIAM_96"):
                if printed_hiam:
                    continue

                dirnames.sort()
                filenames.sort()
                printed_hiam = True

            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level

            f.write(f'{indent}{os.path.basename(dirpath)}/\n')

            if any(ignored in dirpath for ignored in ignore_dirs):
                filenames = []

            sub_indent = ' ' * 4 * (level + 1)
            for filename in filenames:
                f.write(f'{sub_indent}{filename}\n')

        f.write("\nHierarchical directory structure:\n")

        first_level_index = 1  # Counting for level 1 directories, starting from 1
        second_level_index = 1  # Counting for level 2 directories, starting from 1
        third_level_index = 1  # Counting for level 3 directories, starting from 1
        py_file_index = 1  # Numbering for .py files [1] [2] [3]
        yaml_file_letter_seq = 'a'  # Letter counting for .yaml files, starting with 'a'
        special_file_letter_seq = 'A'  # Letter counting for .pkl, .pt, .xlsx files, starting with 'A'
        greek_letter_seq = 'α'  # Greek letter for unmentioned file types

        printed_hiam = False
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if '__pycache__' in dirpath:
                continue

            if not should_print_hiam_directory(dirpath, filenames):
                continue

            if os.path.basename(dirpath).startswith("HIAM_96"):
                if printed_hiam:
                    continue

                dirnames.sort()  # Sort directories by the first and second letters
                filenames.sort()  # Sort filenames
                printed_hiam = True  # Mark as having printed a qualifying HIAM folder

            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level

            if level == 0:
                f.write(f'{os.path.basename(dirpath)}/\n')
                #continue

            if level == 1:
                f.write(f'{first_level_index}. {os.path.basename(dirpath)}/\n')
                first_level_index += 1
                second_level_index = 1
            elif level == 2:
                f.write(f'{indent}{second_level_index}) {os.path.basename(dirpath)}/\n')
                second_level_index += 1
                third_level_index = 1
            elif level == 3:
                f.write(f'{indent}({third_level_index}) {os.path.basename(dirpath)}/\n')
                third_level_index += 1
            else:
                f.write(f'{indent}{os.path.basename(dirpath)}/\n')

            if any(ignored in dirpath for ignored in ignore_dirs):
                filenames = []

            sub_indent = ' ' * 4 * (level + 1)
            for filename in filenames:
                if filename.endswith('.py'):
                    f.write(f'{sub_indent}[{py_file_index}] {filename}\n')
                    py_file_index += 1  # Increment .py file numbering
                elif filename.endswith('.yaml'):
                    f.write(f'{sub_indent}{yaml_file_letter_seq}. {filename}\n')
                    yaml_file_letter_seq = increment_letter(yaml_file_letter_seq)  # Increment .yaml file letters
                elif filename.endswith(('.pkl', '.pt', '.xlsx')):
                    f.write(f'{sub_indent}{special_file_letter_seq}. {filename}\n')
                    special_file_letter_seq = increment_letter(special_file_letter_seq,
                                                               is_upper=True)  # Increment .pkl, .pt, .xlsx file letters
                else:
                    f.write(f'{sub_indent}{greek_letter_seq}. {filename}\n')
                    greek_letter_seq = increment_greek_letter(greek_letter_seq)  # Use Greek letters for other files

# project_root = os.path.abspath(os.path.join(os.getcwd()))
project_root = Find_project_root()
output_file = os.path.join(project_root, 'directory_description.txt')
ignore_dirs = ['.idea', 'DO', '__pycache__', 'DCRNN', 'suzhou - 副本']

# generate_project_structure(project_root, output_file, ignore_dirs)
generate_two_formats_project_structure(project_root, output_file, ignore_dirs)
print("a")