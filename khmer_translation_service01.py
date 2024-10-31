import zmq
import asyncio
import threading
from transformers import VitsModel, AutoTokenizer
import torch
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import io

class KhmerTranslator:
    def __init__(self):
        # Initialize the TTS model and tokenizer
        self.model = VitsModel.from_pretrained("facebook/mms-tts-khm")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-khm")
        self.is_speaking = False  # Flag to indicate if speaking is in progress

        # Setup ZeroMQ context and socket for communication
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:5555")  # Bind to port 5555 to receive messages

        # Initialize asyncio event loop
        self.loop = asyncio.get_event_loop()

    def translate_to_khmer(self, sentence):
        translations = {
            'hello': 'សួស្តី',
            'I': 'ខ្ញុំ',
            'go': 'ទៅ',
            'home': 'ផ្ទះ',
            'like': 'ចូលចិត្ត',
            'today': 'ថ្ងៃនេះ',
            'want': 'ចង់',
            'meet': 'ជួប',
            'you': 'អ្នក',
            'drink': 'ផឹកទឹក'
        }
        return ' '.join(translations.get(word, word) for word in sentence.split())

    def generate_audio(self, sentence):
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors="pt")

        # Generate the audio waveform
        with torch.no_grad():
            output = self.model(**inputs)
            audio_waveform = output.waveform.squeeze().cpu().numpy()

        # Convert the waveform to audio format
        audio_waveform = (audio_waveform * 32767).astype(np.int16)  # Convert to 16-bit PCM format
        audio_segment = AudioSegment(
            audio_waveform.tobytes(),
            frame_rate=self.model.config.sampling_rate,
            sample_width=2,  # 2 bytes for 16-bit PCM
            channels=1  # Mono
        )

        # Save to a byte stream and return
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()

    def play_audio(self, audio_data):
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
        play(audio_segment)

    def speak_khmer_async(self, sentence):
        if self.is_speaking:
            return

        self.is_speaking = True
        def task():
            try:
                # Generate and play audio in a separate thread
                audio_data = self.generate_audio(sentence)
                self.play_audio(audio_data)
            finally:
                self.is_speaking = False

        threading.Thread(target=task, daemon=True).start()

    async def process_translation(self):
        while True:
            sentence = await self.loop.run_in_executor(None, self.socket.recv_string)  # Receive the completed sentence
            khmer_sentence = self.translate_to_khmer(sentence)
            print("Received Sentence:", sentence)
            print("Khmer Translation:", khmer_sentence)
            self.speak_khmer_async(khmer_sentence)

    def run(self):
        # Start the asyncio event loop
        try:
            self.loop.run_until_complete(self.process_translation())
        finally:
            self.loop.close()

if __name__ == "__main__":
    translator = KhmerTranslator()
    translator.run()


# import zmq
# import threading
# from transformers import VitsModel, AutoTokenizer
# import torch
# import sounddevice as sd
# import numpy as np

# class KhmerTranslator:
#     def __init__(self):
#         self.model = VitsModel.from_pretrained("facebook/mms-tts-khm")
#         self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-khm")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.PULL)
#         self.socket.bind("tcp://*:6666")
        
#         self.translations = {
#             'hello': 'សួស្តី', 'I': 'ខ្ញុំ', 'go': 'ទៅ', 'home': 'ផ្ទះ',
#             'like': 'ចូលចិត្ត', 'today': 'ថ្ងៃនេះ', 'want': 'ចង់',
#             'meet': 'ជួប', 'you': 'អ្នក', 'drink': 'ផឹកទឹក'
#         }

#     def translate_to_khmer(self, sentence):
#         return ' '.join(self.translations.get(word, word) for word in sentence.split())

#     def generate_audio(self, sentence):
#         inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             output = self.model(**inputs)
#         return output.waveform.squeeze().cpu().numpy()

#     def play_audio(self, audio_array):
#         sd.play(audio_array, samplerate=self.model.config.sampling_rate)
#         sd.wait()

#     def process_translation(self):
#         while True:
#             sentence = self.socket.recv_string()
#             khmer_sentence = self.translate_to_khmer(sentence)
#             print("Received Sentence:", sentence)
#             print("Khmer Translation:", khmer_sentence)
            
#             audio_array = self.generate_audio(khmer_sentence)
#             threading.Thread(target=self.play_audio, args=(audio_array,)).start()

#     def run(self):
#         print("Khmer Translator is running. Waiting for messages...")
#         self.process_translation()

# if __name__ == "__main__":
#     translator = KhmerTranslator()
#     translator.run()
