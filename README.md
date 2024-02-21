Idea is to build a searchable index for Rick Beato Podcasts.

Inspiration : hubermanlab.com


Functionality:
- an interface to type in search keyword
- list of vides with timestamps where the keyword matches


### Breaking down audio files into smaller chunks
```bash
ffmpeg -i The\ Sting\ Interview-efRQh2vspVc.mp3  -f segment -segment_time 20 -c copy -reset_timestamps 1 -map 0 output_%03d.mp3
```