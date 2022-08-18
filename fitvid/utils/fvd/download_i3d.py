import os
import pathlib
import requests
import torch
from tqdm import tqdm


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit="iB", unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


def download(
    id,
    fname,
    root,
):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination
    URL = "https://drive.google.com/uc?export=download"

    import gdown

    url = f"https://drive.google.com/uc?id={id}"
    output = destination
    gdown.download(url, output, quiet=False)
    # session = requests.Session()
    #
    # response = session.get(URL, params={"id": id}, stream=True)
    # token = get_confirm_token(response)
    #
    # if token:
    #     params = {"id": id, "confirm": token}
    #     response = session.get(URL, params=params, stream=True)
    # save_response_content(response, destination)
    return destination


_I3D_PRETRAINED_ID = "1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT"


def load_i3d_pretrained(device=torch.device("cpu")):
    from fitvid.utils.fvd.pytorch_i3d import InceptionI3d

    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(
        _I3D_PRETRAINED_ID,
        "i3d_pretrained_400.pt",
        pathlib.Path(__file__).parent.resolve(),
    )
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d


if __name__ == "__main__":
    i3d = load_i3d_pretrained()
    import ipdb

    ipdb.set_trace()
