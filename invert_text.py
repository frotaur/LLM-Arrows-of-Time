""" Script to reverse a large file line by line in text mode."""

import argparse

def reverse_file_line_by_line(input_file, output_file):
    """
    Reverse a large file line by line in text mode.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    with open(input_file, "r", encoding="utf-8") as f_in:
        f_in.seek(0, 2)  # Go to the end of the file
        position = f_in.tell()
        current_line = ""

        with open(output_file, "w", encoding="utf-8") as f_out:
            while position >= 0:
                f_in.seek(position)
                try:
                    char = f_in.read(1)
                except UnicodeDecodeError:
                    # In case of a split multi-byte character
                    position -= 1
                    continue

                if char == "\n":
                    f_out.write(current_line[::-1] + "\n")
                    current_line = ""
                else:
                    current_line = char + current_line

                position -= 1

            # Write the last line if there is one
            if current_line:
                f_out.write(current_line[::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Reverse a large file line by line in text mode."""
    )

    parser.add_argument("input_file", type=str, help="""The input file path.""")

    parser.add_argument("output_file", type=str, help="""The output file path.""")

    args = parser.parse_args()

    reverse_file_line_by_line(args.input_file, args.output_file)
