
import os

def fix_app_file():
    file_path = 'app.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Define ranges (using 0-based indexing)
    # Part 1: Lines 0 to 129 (inclusive) -> lines[0:130]
    # Part 2: Lines 130 to 287 (inclusive) -> lines[130:288]
    # Part 3: Lines 288 to 943 (end) -> lines[288:]

    part1 = lines[0:130]
    part2 = lines[130:288]
    part3 = lines[288:]

    # Indent Part 3 by 4 spaces
    indented_part3 = []
    for line in part3:
        if line.strip(): # Don't indent empty lines
            indented_part3.append("    " + line)
        else:
            indented_part3.append(line)

    # Reassemble: Part 1 + Indented Part 3 + Part 2
    new_content = part1 + indented_part3 + part2

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)

    print("File fixed successfully.")

if __name__ == '__main__':
    fix_app_file()

