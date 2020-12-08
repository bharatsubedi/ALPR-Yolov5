# This file contains google utils: https://cloud.google.com/storage/docs/reference/libraries
# pip install --upgrade google-cloud-storage
# from google.cloud import storage

import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def gsutil_getsize(url=''):
    s = subprocess.check_output('gsutil du %s' % url, shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    file = Path(weights).name
    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']  # available models

    if file in models and not os.path.isfile(weights):
        try:  
            url = 'https://github.com/ultralytics/yolov5/releases/download/v3.0/' + file
            print('Downloading %s to %s...' % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('Download error: %s' % e)
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return


def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
    # Downloads a file from Google Drive. from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system('curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s ' % (id, out))
    if os.path.exists('cookie'):  # large file
        s = 'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s' % (get_token(), id, name)
    else:  # small file
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    r = os.system(s)  # execute, capture return
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
