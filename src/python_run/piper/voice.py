import json
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import onnxruntime
from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run

from .config import PhonemeType, PiperConfig
from .const import BOS, EOS, PAD
from .util import audio_float_to_int16

_LOGGER = logging.getLogger(__name__)


@dataclass
class PiperVoice:
    session: onnxruntime.InferenceSession
    config: PiperConfig

    @staticmethod
    def load(
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
    ) -> "PiperVoice":
        """Load an ONNX model and config."""
        if config_path is None:
            config_path = f"{model_path}.json"

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        # set a seed ro reduce randomness
        onnxruntime.set_seed(0)

        providers: List[Union[str, Tuple[str, Dict[str, Any]]]]
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
        else:
            providers = ["CPUExecutionProvider"]

        return PiperVoice(
            config=PiperConfig.from_dict(config_dict),
            session=onnxruntime.InferenceSession(
                str(model_path),
                sess_options=onnxruntime.SessionOptions(),
                providers=providers,
            ),
        )

    def phonemize(self, text: str) -> List[List[str]]:
        """Text to phonemes grouped by sentence."""
        if self.config.phoneme_type == PhonemeType.ESPEAK:
            if self.config.espeak_voice == "ar":
                # Arabic diacritization
                # https://github.com/mush42/libtashkeel/
                text = tashkeel_run(text)

            return phonemize_espeak(text, self.config.espeak_voice)

        if self.config.phoneme_type == PhonemeType.TEXT:
            return phonemize_codepoints(text)

        raise ValueError(f"Unexpected phoneme type: {self.config.phoneme_type}")

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Phonemes to ids."""
        id_map = self.config.phoneme_id_map
        ids: List[int] = list(id_map[BOS])

        for phoneme in phonemes:
            if phoneme not in id_map:
                _LOGGER.warning("Missing phoneme from id map: '%s'", phoneme)
                continue

            ids.extend(id_map[phoneme])
            ids.extend(id_map[PAD])

        ids.extend(id_map[EOS])

        return ids

    def synthesize_with_alignment_data(
            self,
            text: str,
            wav_file: wave.Wave_write,
            alignment_file: str,
            phoneme_input=False,
            speaker_id: Optional[int] = None,
            length_scale: Optional[float] = None,
            noise_scale: Optional[float] = None,
            noise_w: Optional[float] = None,
            sentence_silence: float = 0.0,
            correction_factor: Optional[float] = 0.66
    ):
        # 1. split text in sentences
        # 2. split sentence in words
        # 3. build word alignment data for sentence
        # 4. synthesize sentence
        # 5. write sentence to wave-file
        # 6. go back to 1

        sentences = text.split(".")[:-1]
        print(f"{sentences=}")
        sentences = [s + "." for s in sentences]

        with open(alignment_file, "w"):
            pass

        global_time = 0
        global_offset = 0

        with wave.open("single_word.wav", "wb") as word_wav_file:

            word_wav_file.setframerate(self.config.sample_rate)
            word_wav_file.setsampwidth(2)  # 16-bit
            word_wav_file.setnchannels(1)  # mono

            for sentence in sentences:
                # (word, start_time, end_time, start_offset, phonemes)
                word_alignment = [[s,  0, 0, 0, []] for s in sentence.strip().split(" ")]

                accumulated_time = 0
                offset = 0
                for index, word_data in enumerate(word_alignment):
                    for audio_bytes in self.synthesize_stream_raw(
                       text=word_data[0],
                       speaker_id=speaker_id,
                       length_scale=length_scale,
                       noise_scale=noise_scale,
                       noise_w=noise_w,
                    ):
                        num_silence_samples = int(sentence_silence * self.config.sample_rate)
                        silence_bytes = bytes(num_silence_samples * 2)

                        word_wav_file.writeframes(audio_bytes)

                        word_alignment[index][1] = accumulated_time
                        word_alignment[index][3] = offset

                        length = len(audio_bytes) / (self.config.sample_rate * 2) * correction_factor

                        accumulated_time += length
                        offset += len(word_data[0]) + 1

                        word_alignment[index][2] = accumulated_time

                        sentence_data = word_alignment[index][0]
                        start_time_data = word_alignment[index][1] + global_time
                        end_time_data = word_alignment[index][2] + global_time
                        offset_data = word_alignment[index][3] + global_offset

                        print(f"{start_time_data}, {end_time_data}, {sentence_data.rstrip()}, {offset_data}")
                        with open(alignment_file, "a") as file:
                            file.write(f"{start_time_data}, {end_time_data}, {sentence_data.rstrip()}, {offset_data}\n")

                for audio_bytes in self.synthesize_stream_raw(
                       sentence,
                       speaker_id=speaker_id,
                       length_scale=length_scale,
                       noise_scale=noise_scale,
                       noise_w=noise_w,
                       sentence_silence=sentence_silence,
                ):
                    global_time += len(audio_bytes) / (self.config.sample_rate * 2)
                    global_offset += len(sentence)
                    print(f"sentence: {len(audio_bytes) / (self.config.sample_rate * 2)}, {len(sentence)}")
                    print(f"{global_time=}, {global_offset=}")
                    wav_file.writeframes(audio_bytes)

    def synthesize(
        self,
        text: str,
        wav_file: wave.Wave_write,
        alignment_file=None,
        phoneme_input=False,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: float = 0.0,
        correction_factor: Optional[float] = 0.66
    ):
        """Synthesize WAV audio from text."""
        wav_file.setframerate(self.config.sample_rate)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setnchannels(1)  # mono

        if alignment_file is not None:
            self.synthesize_with_alignment_data(
                text,
                wav_file=wav_file,
                alignment_file=alignment_file,
                phoneme_input=phoneme_input,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                sentence_silence=sentence_silence,
                correction_factor=correction_factor
            )

        else:
            for audio_bytes in self.synthesize_stream_raw(
                    text,
                    speaker_id=speaker_id,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w=noise_w,
                    sentence_silence=sentence_silence,
            ):
                wav_file.writeframes(audio_bytes)

    def synthesize_stream_raw(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: float = 0.0,
    ) -> Iterable[bytes]:
        """Synthesize raw audio per sentence from text."""
        sentence_phonemes = self.phonemize(text)

        num_silence_samples = int(sentence_silence * self.config.sample_rate)
        silence_bytes = bytes(num_silence_samples * 2)

        for phonemes in sentence_phonemes:
            phoneme_ids = self.phonemes_to_ids(phonemes)
            yield self.synthesize_ids_to_raw(
                phoneme_ids,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            ) + silence_bytes

    def synthesize_stream_raw_from_phonemes(
        self,
        text_phonemes: str,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: float = 0.0
    ) -> Iterable[bytes]:
        """
        Synthesize speech directly from phonemes
        """
        # 16-bit mono
        num_silence_samples = int(sentence_silence * self.config.sample_rate)
        silence_bytes = bytes(num_silence_samples * 2)

        sentences = text_phonemes.split("\n")
        print(sentences)
        sentence_phonemes = []
        for sentence in sentences:
            if sentence == '':
                continue
            sentence_phonemes.append(list(sentence))

        for phonemes in sentence_phonemes:
            phoneme_ids = self.phonemes_to_ids(phonemes)
            yield self.synthesize_ids_to_raw(
                phoneme_ids,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            ) + silence_bytes

    def synthesize_ids_to_raw(
        self,
        phoneme_ids: List[int],
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> bytes:
        """Synthesize raw audio from phoneme ids."""
        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w is None:
            noise_w = self.config.noise_w

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32,
        )

        if (self.config.num_speakers > 1) and (speaker_id is None):
            # Default speaker
            speaker_id = 0

        sid = None

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)

        # Synthesize through Onnx
        audio = self.session.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))
        audio = audio_float_to_int16(audio.squeeze())

        return audio.tobytes()

