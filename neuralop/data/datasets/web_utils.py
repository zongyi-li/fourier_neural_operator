"""
Utils for downloading dataset files from the web.

Mostly borrowed from torchvision.datasets.utils
"""

import hashlib
import logging
import os
from pathlib import Path
import requests
import sys
from typing import Any, Optional, Union, List
import tarfile

logger = logging.Logger(name=__name__)
logger.setLevel(logging.root.level)

# md5 utils from torchvision.datasets.utils


def calculate_md5(fpath: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: Union[str, Path], md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: Union[str, Path], md5: Optional[str] = None) -> bool:
    if isinstance(fpath, str):
        fpath = Path(fpath)
    if not fpath.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_from_url(
    url: str,
    root: Union[str, Path],
    filename: Optional[Union[str, Path]] = None,
    md5: Optional[str] = None,
    size: Optional[int] = None,
    chunk_size: Optional[int] = 256 * 64,
    extract_tars: bool = True,
) -> None:
    """download_from_url downloads a file from a url with
    an optional md5 checksum.

    If the file is a tarball, optionally extract.

    Parameters
    ----------
    url : str
        url where dataset archive file is stored
    root : Union[str, Path]
        root of dataset folder on local machine
    filename : Optional[Union[str, Path]], optional
        name into which to download, by default None
    md5 : Optional[str], optional
        md5 checksum to verify file correctness, by default None
    size: Optional[int], optional
        size of the file expected in bytes, to check
    chunk_size : int, optional
        size of chunks to use when streaming files
    extract_tars: bool, optional
        whether to extract .tgz archives, by default True
    """
    if isinstance(root, str):
        root = Path(root)

    root = root.expanduser()
    if not filename:
        # grab file ext from basename
        filename = url.split("/")[-1]
    fpath = root / filename

    root.mkdir(parents=True, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        logger.info("Using downloaded and verified file: " + str(fpath))
        return

    # download and stream file in chunks
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fpath, "wb") as f:
            curr_size = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    # write file, flush and fsync to prevent memory bloat
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    # print progress dynamically
                    curr_size += chunk_size
                    prog = curr_size / size
                    print(f"Download in progress: {prog:.2%}".ljust(40), end="\r")

            assert (
                fpath.stat().st_size == size
            ), (
                f"Error: mismatch between expected and true size of downloaded file {fpath}. "
                "Delete the file and try again."
            )

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
    else:
        logger.info("saved to {fpath} successfully")
        # extract tarfiles to their given filenames on root
        if extract_tars:
            logger.info("Extracting {fpath}...")
            archive = tarfile.open(fpath)
            archive.extractall(path=root)
            name_str = ", ".join(archive.getnames())
            logger.info(f"Extracted {name_str}")


def download_from_zenodo_record(
    record_id: str,
    root: Union[str, Path],
    files_to_download: Optional[List[str]] = None,
):
    """download_from_zenodo_record downloads files in "files_to_download"
    from zenodo record with ID record_id

    Parameters
    ----------
    record_id : str
        ID of record in Zenodo's database
    root : Union[str, Path]
        root directory into which to download archive
    files_to_download : Optional[List[str]]
        list of filenames to download from record.
        If None, downloads all. Default None
    """
    if files_to_download is not None and len(files_to_download) == 0:
        return
    zenodo_api_url = "https://zenodo.org/api/records/"
    url = f"{zenodo_api_url}{record_id}"
    resp = requests.get(url)
    assert (
        resp.status_code == 200
    ), f"Error: request failed with status code {resp.status_code}"
    response_json = resp.json()
    for file_record in response_json["files"]:
        fname = file_record["key"]
        if files_to_download is None or fname in files_to_download:
            download_from_url(
                url=file_record["links"]["self"],
                md5=file_record["checksum"][4:],  # md5 stored as 'md5:xxxxx'
                size=file_record["size"],
                root=root,
                filename=fname,
                extract_tars=True,
            )
