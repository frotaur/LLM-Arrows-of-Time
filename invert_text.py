


def reverse_file_line_by_line(input_file, output_file):
    """
    Reverse a large file line by line in text mode.

    :param input_file: Path to the input file.
    :param output_file: Path to the output file.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in:
        f_in.seek(0, 2)  # Go to the end of the file
        position = f_in.tell()
        current_line = ''

        with open(output_file, 'w', encoding='utf-8') as f_out:
            while position >= 0:
                f_in.seek(position)
                try:
                    char = f_in.read(1)
                except UnicodeDecodeError:
                    # In case of a split multi-byte character
                    position -= 1
                    continue

                if char == '\n':
                    f_out.write(current_line[::-1] + '\n')
                    current_line = ''
                else:
                    current_line = char + current_line

                position -= 1

            # Write the last line if there is one
            if current_line:
                f_out.write(current_line[::-1])



def reverse_file_fast(input_file, output_file,chunk_size=70):
    """
    Reverse a large file line by line in text mode.

    :param input_file: Path to the input file.
    :param output_file: Path to the output file.
    """
    with open(input_file, 'rb') as f_in:
        f_in.seek(0, 2)  # Go to the end of the file
        position = f_in.tell()

        with open(output_file,'w',encoding='utf-8') as f_out:
            while position >0 :
                oldpos = position
                position = max(0, oldpos-chunk_size)
                notdone=True
                while notdone:
                    try:
                        f_in.seek(position)
                        chunk = f_in.read(oldpos-position) # thiS CUNT read CHARACTERS, not BYTES
                        notdone=False
                        chunk = chunk.decode('utf-8')
                    except UnicodeDecodeError:
                        # In case of a split multi-byte character
                        position =max(0, position-1)
                        notdone=True
            
                f_out.write(chunk[::-1])


# Usage
reverse_file_fast('fr.txt', 'fr_R.txt')

# reverse_file_line_by_line_fast('test_R.txt','test_RR.txt')