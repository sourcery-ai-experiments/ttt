# ttt
Basically an addon to trunk-recorder which runs the audio through whisper to get text and then happily forwards anywhere apprise services.

But much more documentation after everything actually works.

## destinations.csv
This is a pretty basic, 3 column csv with your system shortname, the talkgroup id in Decimal, and the apprise url to report to. 

```csv
short_name,talkgroup,url
pwcpw25,1003,ntfy://notarealtopic
```