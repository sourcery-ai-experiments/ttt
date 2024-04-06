#!/usr/bin/env python

import json
import os
import time
from datetime import datetime
from pathlib import Path

import apprise
import requests


def transcribe_call(destinations):
    # First lets search the media directory for all json, sorted by creation time
    jsonlist = sorted(
        Path("media/transcribe").rglob("*.[jJ][sS][oO][nN]"), key=os.path.getctime
    )

    # If the queue is empty, pause for 5 seconds and then restart polling
    if not jsonlist:
        print("Empty queue. Sleep 5 seconds and check again.")
        time.sleep(5)
        return ()

    for jsonfile in jsonlist:
        # Ok, let's grab the first json and pull it out and then the matching wav file
        audiofile = Path(jsonfile).with_suffix(".wav")

        print(f"Processing: {audiofile}")

        # Now load the actual json data into calljson
        calljson = jsonfile.read_text()
        calljson = json.loads(calljson)

        # Send the json and audiofile to a function to transcribe
        # If TTT_DEEPGRAM_KEY is set, use deepgram, else
        # if TTT_WHISPER_URL is set, use whisper.cpp else
        # fasterwhisper deepgram_key := whispercpp_url :=
        if os.environ.get("TTT_DEEPGRAM_KEY", False):
            calljson = transcribe_deepgram(calljson, audiofile)
        elif os.environ.get("TTT_WHISPERCPP_URL", False):
            calljson = transcribe_whispercpp(calljson, audiofile)
        else:
            calljson = transcribe_fasterwhisper(calljson, audiofile)

        # Ok, we have text back, send for notification
        send_notifications(calljson, destinations)

        # And now delete the files from the transcribe directory
        Path.unlink(jsonfile)
        Path.unlink(audiofile)


def transcribe_whispercpp(calljson, audiofile):
    whisper_url = os.environ.get("TTT_WHISPERCPP_URL", "http://whisper:8080")

    # Now send the files over to whisper for transcribing
    files = {
        "file": (None, audiofile.read_bytes()),
        "temperature": (None, "0.0"),
        "temperature_inc": (None, "0.2"),
        "response_format": (None, "json"),
    }

    try:
        response = requests.post(f"{whisper_url}/inference", files=files)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        raise Exception("A request error occurred.")

    calltext = response.json()

    # And now merge that dict into calljson so [text] in calljson is the transcript
    calljson = {**calljson, **calltext}
    return calljson


def transcribe_fasterwhisper(calljson, audiofile):
    from faster_whisper import WhisperModel

    model_size = os.environ.get(
        "TTT_FASTERWHISPER_MODEL_SIZE", "Systran/faster-distil-whisper-large-v3"
    )
    device = os.environ.get("TTT_FASTERWHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("TTT_FASTERWHISPER_COMPUTE_TYPE", "int8")
    vad_filter = os.environ.get("TTT_FASTERWHISPER_VAD_FILTER", False)

    model = WhisperModel(
        model_size, device=device, compute_type=compute_type, download_root="models"
    )

    # This whisper wants the path, not bytes but we need to cast it from pathlib to str
    audiofile = str(audiofile)
    segments, info = model.transcribe(audiofile, beam_size=5, vad_filter=vad_filter)

    calltext = "".join(segment.text for segment in segments)

    calljson["text"] = calltext

    return calljson


def transcribe_deepgram(calljson, audiofile):
    deepgram_key = os.environ.get("TTT_DEEPGRAM_KEY")
    headers = {
        "Authorization": f"Token {deepgram_key}",
        "Content-Type": "audio/wav",
    }
    params = {
        "model": "nova-2-phonecall",
        "smart_format": "true",
        "numerals": "true",
    }

    data = audiofile.read_bytes()
    try:
        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            data=data,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return

    json = response.json()

    # We take the json returned from deepgram and pull out the "transcript"
    # then tack it onto the calljson dict as "text" which is what whisper
    # normally uses
    calltext = json["results"]["channels"][0]["alternatives"][0]["transcript"]
    calljson["text"] = calltext
    return calljson


def send_notifications(calljson, destinations):
    body = calljson["text"]
    title = (
        calljson["talkgroup_description"]
        + " @ "
        + str(datetime.fromtimestamp(calljson["start_time"]))
    )

    talkgroup = calljson["talkgroup"]
    notify_url = destinations[talkgroup]

    apobj = apprise.Apprise()
    apobj.add(notify_url)
    apobj.notify(
        body=body,
        title=title,
    )


def import_notification_destinations():
    # I didn't really want to add a pandas dependency, but it did what I want in one
    # line so that's hard to argue with
    import pandas as pd

    return pd.read_csv("destinations.csv", index_col=0).squeeze().to_dict()


def main():
    destinations = import_notification_destinations()
    while 1:
        transcribe_call(destinations)


if __name__ == "__main__":
    main()
