#!/usr/bin/env python

import json
import os
import time
from datetime import datetime
from pathlib import Path

import apprise
import requests
from better_profanity import profanity


def transcribe_call(destinations):
    # First lets search the media directory for all json, sorted by creation time
    jsonlist = sorted(
        Path("media/transcribe").rglob("*.[jJ][sS][oO][nN]"), key=os.path.getctime
    )

    # If the queue is empty, pause for 5 seconds and then restart polling
    if not jsonlist:
        print("Empty queue. Sleep 5 seconds and check again.")
        time.sleep(5)
        switch_model("normal")
        return ()

    for jsonfile in jsonlist:
        # Ok, let's grab the first json and pull it out and then the matching wav file
        audiofile = Path(jsonfile).with_suffix(".wav")

        print(f"Processing: {audiofile}")

        # Now load the actual json data into calljson
        calljson = jsonfile.read_text()
        calljson = json.loads(calljson)

        # Check if we are running behind
        queue_time = float(datetime.now().timestamp()) - calljson["start_time"]
        if queue_time > 180:
            print("Queue exceeds 3 minutes")
            switch_model("quick")

        # Now send the files over to whisper for transcribing
        files = {
            "file": (None, audiofile.read_bytes()),
            "temperature": (None, "0.0"),
            "temperature_inc": (None, "0.2"),
            "response_format": (None, "json"),
        }

        try:
            response = requests.post("http://10.0.1.200:8888/inference", files=files)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

        calltext = response.text

        # Whisper seems to think radio conversation is a bit more colorful than it
        # normally is.  Let's try and make it more PG
        calltext = profanity.censor(calltext)

        # Load the json from whisper into a json/dict
        calltext = json.loads(calltext)

        # And now merge that dict into calljson so [text] in calljson is the transcript
        calljson = {**calljson, **calltext}

        # Ok, we have text back, send for notification
        send_notifications(calljson, destinations)

        # And now delete the files from the transcribe directory
        Path.unlink(jsonfile)
        Path.unlink(audiofile)


def send_notifications(calljson, destinations):
    talkgroup_description = calljson["talkgroup_description"]
    talkgroup = calljson["talkgroup"]
    compiledcall = calljson["text"]

    notify_url = destinations[talkgroup]

    apobj = apprise.Apprise()
    apobj.add(notify_url)
    apobj.notify(
        body=compiledcall,
        title=talkgroup_description,
    )


def switch_model(model):
    # If we're running behind, switch to the small model (quick)
    # If we catch up and the queue is zero, go back to medium. (normal)
    # THIS REQUIRES THE MODELS LOCALLY TO THIS FILE SINCE THEY WILL BE UPLOADED
    # FROM HERE.
    if model == "quick":
        files = {  #
            "model": (None, open("models/ggml-small.en.bin", "rb")),
        }
    else:
        files = {
            "model": (None, open("models/ggml-medium.en.bin", "rb")),
        }

    try:
        requests.post("http://10.0.1.200:8888/load", files=files)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def import_notification_destinations():
    # I didn't really want to add a pandas dependency, but it did what I want in one
    # line so that's hard to argue with
    import pandas as pd

    return pd.read_csv("destinations.csv", index_col=0).squeeze().to_dict()


def main():
    profanity.load_censor_words()
    destinations = import_notification_destinations()
    while 1:
        transcribe_call(destinations)


if __name__ == "__main__":
    main()
