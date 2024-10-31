import zmq
import asyncio
import signal
from transformers import VitsModel, AutoTokenizer
import torch
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import io
from functools import lru_cache

class KhmerTranslator:
    def __init__(self):
        self.model = VitsModel.from_pretrained("facebook/mms-tts-khm")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-khm")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.audio_queue = asyncio.Queue()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:5555")
        self.loop = asyncio.get_event_loop()
        self.translation_cache = {}
        self.running = True

    def translate_to_khmer(self, sentence):
        if sentence in self.translation_cache:
            return self.translation_cache[sentence]
        translations = {
            'hello': 'សួស្តី', 'I': 'ខ្ញុំ', 'go': 'ទៅ', 'home': 'ផ្ទះ',
            'like': 'ចូលចិត្ត', 'today': 'ថ្ងៃនេះ', 'want': 'ចង់',
            'meet': 'ជួប', 'you': 'អ្នក', 'drink': 'ផឹកទឹក', 'how are you?': 'តើអ្នកសុខសប្បាយទេ?',
            'toilet':'បន្ទប់ទឹក', 'all of you': 'អ្នកទាំងអស់គ្នា', 'again':'ម្តងទៀត', 'sad':'ពិបាកចិត្ត'
        }
        translated = ' '.join(translations.get(word, word) for word in sentence.split())
        self.translation_cache[sentence] = translated
        return translated

    @lru_cache(maxsize=100)
    def generate_audio(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            audio_waveform = output.waveform.squeeze().cpu().numpy()
        audio_waveform = (audio_waveform * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_waveform.tobytes(),
            frame_rate=self.model.config.sampling_rate,
            sample_width=2,
            channels=1
        )
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()

    async def play_audio_async(self, audio_data):
        def play_in_thread():
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            play(audio_segment)
        await self.loop.run_in_executor(None, play_in_thread)

    async def audio_player(self):
        while self.running:
            try:
                audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                await self.play_audio_async(audio_data)
                self.audio_queue.task_done()
            except asyncio.TimeoutError:
                continue

    async def process_translation(self):
        while self.running:
            try:
                sentence = await self.loop.run_in_executor(None, self.socket.recv_string)
                khmer_sentence = await self.loop.run_in_executor(None, self.translate_to_khmer, sentence)
                print("Received Sentence:", sentence)
                print("Khmer Translation:", khmer_sentence)
                audio_data = await self.loop.run_in_executor(None, self.generate_audio, khmer_sentence)
                await self.audio_queue.put(audio_data)
            except zmq.ZMQError:
                if not self.running:
                    break
                else:
                    raise

    def stop(self):
        print("Stopping Khmer Translator...")
        self.running = False
        self.socket.close()
        self.context.term()

    async def run_async(self):
        audio_task = self.loop.create_task(self.audio_player())
        translation_task = self.loop.create_task(self.process_translation())
        
        try:
            await asyncio.gather(audio_task, translation_task)
        except asyncio.CancelledError:
            pass

    def run(self):
        def signal_handler(sig, frame):
            print("Interrupt received, stopping...")
            self.loop.stop()
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("Khmer Translator is running. Press Ctrl+C to stop.")
        try:
            self.loop.run_until_complete(self.run_async())
        finally:
            self.loop.close()

if __name__ == "__main__":
    translator = KhmerTranslator()
    translator.run()