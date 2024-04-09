#!/usr/bin/env python

import json
import os
import time
from datetime import datetime
from pathlib import Path

import apprise
import requests
import scrubadub
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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
        # fasterwhisper
        if os.environ.get("TTT_DEEPGRAM_KEY", False):
            calljson = transcribe_deepgram(calljson, audiofile)
        elif os.environ.get("TTT_WHISPERCPP_URL", False):
            calljson = transcribe_whispercpp(calljson, audiofile)
        else:
            calljson = transcribe_transformers(calljson, audiofile)

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
        raise Exception("A request error occurred.") from e

    calltext = response.json()

    # And now merge that dict into calljson so [text] in calljson is the transcript
    calljson = {**calljson, **calltext}
    return calljson


def transcribe_transformers(calljson, audiofile):
    audiofile = str(audiofile)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audiofile)
    calljson["text"] = result["text"]
    return calljson


def transcribe_deepgram(calljson, audiofile):
    deepgram_key = os.environ.get("TTT_DEEPGRAM_KEY")
    headers = {
        "Authorization": f"Token {deepgram_key}",
        "Content-Type": "audio/wav",
    }
    params = {
        "model": "whisper-large",
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
    # Scrubadub redacts PII let's try and clean the text before
    # goes out the door
    scrubber = scrubadub.Scrubber()
    scrubber.remove_detector("email")
    body = scrubber.clean(calljson["text"])
    title = (
        calljson["talkgroup_description"]
        + " @ "
        + str(datetime.fromtimestamp(calljson["start_time"]))
    )

    talkgroup = str(calljson["talkgroup"])
    notify_url = destinations[talkgroup]

    apobj = apprise.Apprise()
    apobj.add(notify_url)
    apobj.notify(
        body=body,
        title=title,
    )


def import_notification_destinations():
    import csv

    destinations = {}
    with open("destinations.csv", mode="r") as inp:
        reader = csv.reader(inp)
        destinations = {rows[0]: rows[1] for rows in reader}

    return destinations


def main():
    # Import the apprise destinations to send calls
    destinations = import_notification_destinations()

    while 1:
        transcribe_call(destinations)


if __name__ == "__main__":
    main()
