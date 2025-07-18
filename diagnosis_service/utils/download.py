import tempfile

import requests


def download_file(url: str, suffix: str) -> str:
    """
    Download a file from the given URL and return the local file path.

    Parameters:
        url : str
            The URL of the file to download.
        suffix : str
            The suffix to use for the temporary file (e.g., '.wav').

    Returns:
        str : The local file path where the downloaded file is saved.
    """
    response = requests.get(url, timeout=600)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(response.content)
        return temp_file.name
