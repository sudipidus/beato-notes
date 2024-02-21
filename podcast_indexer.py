import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import youtube_dl
import psycopg2, ffmpeg

class PodcastIndexer:
    def __init__(self, url):
        self.url = url
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def download_audio(self):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'output.mp3',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])

    def split_audio(self):
        (   
            ffmpeg
            .input('output.mp3')
            .output('output_%03d.mp3', f='segment', segment_time=60)
            .run()
        )

    def transcribe_audio(self, filename):
        result = self.pipe(filename)
        return result["chunks"]

    def save_to_database(self, results):
        conn = psycopg2.connect(
            dbname="beato_notes",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5435"
        )
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS podcasts (id SERIAL PRIMARY KEY, url VARCHAR, timestamp_start REAL, timestamp_end REAL, text TEXT)")
        for chunk in results:
            timestamp_start = chunk['timestamp'][0]
            timestamp_end = chunk['timestamp'][1]
            text = chunk['text']
            cur.execute("INSERT INTO podcasts (url, timestamp_start, timestamp_end, text) VALUES (%s, %s, %s, %s)", (self.url, timestamp_start, timestamp_end, text))
        conn.commit()
        cur.close()
        conn.close()

    def process_podcast(self):
        self.download_audio()
        self.split_audio()
        results = self.transcribe_audio('output_000.mp3')
        self.save_to_database(results)


if __name__=='__main__':
    # Example usage:
    podcast_indexer = PodcastIndexer("https://www.youtube.com/watch?v=FwUiiUuNCM0")
    podcast_indexer.process_podcast()
