def file_reader(filepath: str) -> str:
    """Reads the text file from given location and append all lines with a space

    Parameters
    ---------
    filepath : str
        Path to the text file

    Returns
    -------
    docstring : str
        String representation of the text file
    """
    doc = []
    with open(filepath, 'r') as file:
        doc = file.readlines()

    docstring = ''

    if len(doc) == 1:
        return doc[0]

    for line in doc:
        line = line.rstrip('\n')
        if line != '':
            docstring += line + ' '

    return docstring


def file_writer(filepath: str, doc: str):
    """Writes content to file

    Parameters
    ----------
    filepath : str
        Path to the text file

    doc : str
        Content to be written in the text file
    """
    with open(filepath, 'w') as file:
        file.write(doc)

    return
