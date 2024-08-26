import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from glob import glob
from .augmentation import augment_raw_audio

class ICBHIDataset(Dataset):
    def __init__(self, data_folder, audio_conf, mode='train'):
        """
        ICBHI Dataset for AudioMAE
        :param data_folder: folder containing WAV files
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param mode: 'train' or 'test'
        """
        self.data_folder = data_folder
        self.audio_conf = audio_conf
        self.mode = mode
        print(f'---------------the {self.mode} dataloader---------------')
        
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        self.mixup = self.audio_conf.get('mixup')
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.noise = self.audio_conf.get('noise')
        self.sample_rate = self.audio_conf.get('sample_rate', 16000)
        
        print(f'using following mask: {self.freqm} freq, {self.timem} time')
        print(f'using mix-up with rate {self.mixup}')
        print(f'Dataset: {self.dataset}, mean {self.norm_mean:.3f} and std {self.norm_std:.3f}')
        
        self.data = sorted(glob(os.path.join(self.data_folder, '*.wav')))
        self.label_num = 2  # ICBHI has 2 classes: normal and abnormal
        print(f'number of classes: {self.label_num}')
        print(f'number of files: {len(self.data)}')

    def _get_label_from_filename(self, filename):
        # Assuming the label is encoded in the filename
        # Modify this function according to your filename format
        if 'normal' in filename.lower():
            return 0
        else:
            return 1

    def _wav2fbank(self, filename, filename2=None):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        # Resample if necessary
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # mixup
        if filename2:
            waveform2, sr2 = torchaudio.load(filename2)
            waveform2 = waveform2 - waveform2.mean()
            if sr2 != self.sample_rate:
                waveform2 = torchaudio.functional.resample(waveform2, sr2, self.sample_rate)
            
            if waveform.shape[1] != waveform2.shape[1]:
                if waveform.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0, 0:waveform.shape[1]]
            
            mix_lambda = np.random.beta(10, 10)
            waveform = mix_lambda * waveform + (1 - mix_lambda) * waveform2

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.sample_rate, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank, mix_lambda if filename2 else 0

    def __getitem__(self, index):
        filename = self.data[index]
        label = self._get_label_from_filename(filename)
        
        # apply mix-up
        if random.random() < self.mixup and self.mode == 'train':
            mix_sample_idx = random.randint(0, len(self.data) - 1)
            filename2 = self.data[mix_sample_idx]
            fbank, mix_lambda = self._wav2fbank(filename, filename2)
            label = mix_lambda * label + (1 - mix_lambda) * self._get_label_from_filename(filename2)
        else:
            fbank, _ = self._wav2fbank(filename)

        # SpecAug for training
        if self.mode == 'train':
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = fbank.transpose(0, 1).unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            fbank = torch.transpose(fbank.squeeze(), 0, 1)

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        
        if self.noise and self.mode == 'train':
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        return fbank, label

    def __len__(self):
        return len(self.data)