import urllib.request
import tarfile

from tqdm import tqdm


# For progress bar
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def fetch_pascal():
    # Downloading PASCAL data
    url_pascal = (
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    )
    download_url(url_pascal, "./pascal.tar")

    with tarfile.open("./pascal.tar", "r") as tar_file:
        tar_file.extractall("./data")


if __name__ == "__main__":
    fetch_pascal()
