# File: modality_correlation/correlation_dataset.py
import torch
import random
from torch.utils.data.dataset import Dataset
import numpy as np
import os, sys

current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

from src.dataset import Multimodal_Datasets

class UnifiedMultimodalDataset(Multimodal_Datasets):
    """
    Unified Dataset:
    - Inherits from Multimodal_Datasets in src/dataset.py, retaining methods like get_dim() / get_seq_len().
    - Decides whether to return a regular sample or (positive, negative) sample based on for_correlation.
    - Decides whether to perturb the current sample (A/B/C) based on perturbation_ratio in __getitem__.
    """

    def __init__(self,
                 dataset_path,
                 data='mosei_senti',
                 split_type='train',
                 if_align=False,
                 max_samples=None,
                 for_correlation=False,
                 perturbation_ratio=0.0,
                 noise_std=0.05,
                 strategy_weights=(1/3, 1/3, 1/3),
                 # NEW: negative generation mode
                 neg_mode='independent',  # 'independent' (recommended) or 'original'
                 neg_strategy_weights=(0.6, 0.2, 0.2),  # A/B/C weights for each modality neg
                 ):
        """
        Args:
            for_correlation:
                If True, returns ((meta, text, audio, vision), (text_neg, audio_neg, vision_neg), label, META)
                Used for correlation pretraining.
                If False, returns normal sample.
            perturbation_ratio:
                Prob to perturb the *positive* sample using apply_perturbation (A/B/C).
            noise_std:
                Gaussian noise strength used by strategy C.
            strategy_weights:
                Relative probabilities for apply_perturbation A/B/C.
            neg_mode:
                - 'independent' : Generate text_neg/audio_neg/vision_neg independently (fixes neg==pos degeneration).
                - 'original'    : Keep your previous behavior (replace only one modality / shift all / noise all).
            neg_strategy_weights:
                When neg_mode='independent', per-modality choose A/B/C with these weights.
                Default: more often cross-sample replacement (A), sometimes shift (B) or noise (C).
        """
        super(UnifiedMultimodalDataset, self).__init__(dataset_path, data, split_type, if_align, max_samples)

        self.for_correlation = for_correlation
        self.perturbation_ratio = perturbation_ratio
        self.noise_std = noise_std

        # for apply_perturbation (positive sample perturbation)
        self.strategies = ['A', 'B', 'C']
        self.strategy_weights = list(strategy_weights)

        # for negative generation in correlation pretraining
        self.neg_mode = neg_mode
        self.neg_strategy_weights = list(neg_strategy_weights)

    def __getitem__(self, index):
        """
        If for_correlation=False:
            Returns: ((meta, text, audio, vision), label, (meta,))
        If for_correlation=True:
            Returns: ((meta, text, audio, vision), (text_neg, audio_neg, vision_neg), label, META)
        """
        (meta, text, audio, vision), label, META = super(UnifiedMultimodalDataset, self).__getitem__(index)

        # Optional perturbation on the POSITIVE sample
        if random.random() < self.perturbation_ratio:
            text, audio, vision, label = self.apply_perturbation(index, text, audio, vision, label)

        if not self.for_correlation:
            return ((meta, text, audio, vision), label, (meta,))
        else:
            text_pos = text.clone()
            audio_pos = audio.clone()
            vision_pos = vision.clone()

            text_neg, audio_neg, vision_neg = self.generate_negative_sample(
                index, text_pos, audio_pos, vision_pos
            )

            return ((meta, text_pos, audio_pos, vision_pos),
                    (text_neg, audio_neg, vision_neg),
                    label,
                    META)

    # -----------------------
    # Positive perturbations
    # -----------------------
    def apply_perturbation(self, index, text, audio, vision, label):
        """
        Apply a random perturbation (A/B/C) to the *positive* sample.
        """
        chosen_strategy = random.choices(self.strategies, weights=self.strategy_weights, k=1)[0]

        text_out = text.clone()
        audio_out = audio.clone()
        vision_out = vision.clone()
        final_label = label.clone()

        if chosen_strategy == 'A':
            # Strategy A: Randomly replace ONE modality from another sample (kept as your original behavior)
            chosen_modality = random.choice(['T', 'A', 'V'])
            rand_idx = self._sample_other_index(exclude=index)

            other_text = self.text[rand_idx]
            other_audio = self.audio[rand_idx]
            other_vision = self.vision[rand_idx]
            other_label = self.labels[rand_idx]

            if chosen_modality == 'T':
                text_out = other_text.clone()
            elif chosen_modality == 'A':
                audio_out = other_audio.clone()
            else:
                vision_out = other_vision.clone()

            # If labels differ, average them (your original logic)
            final_label = 0.5 * (label + other_label)

        elif chosen_strategy == 'B':
            # Strategy B: Time shift (all modalities)
            text_out = self.shift_sequence(text_out)
            audio_out = self.shift_sequence(audio_out)
            vision_out = self.shift_sequence(vision_out)

        elif chosen_strategy == 'C':
            # Strategy C: noise to audio/vision + replace one token in text
            audio_out = audio_out + torch.randn_like(audio_out) * self.noise_std
            vision_out = vision_out + torch.randn_like(vision_out) * self.noise_std
            if text_out.size(0) > 0:
                idx_word = random.randint(0, text_out.size(0) - 1)
                text_out[idx_word] = torch.randn_like(text_out[idx_word]) * self.noise_std

        return text_out, audio_out, vision_out, final_label

    def shift_sequence(self, seq):
        """
        Shift operation for strategy B.
        """
        if seq.size(0) > 1:
            shifted = torch.zeros_like(seq)
            shifted[:-1] = seq[1:]
            return shifted
        else:
            return seq

    # -----------------------
    # Negative generation (Correlation pretraining)
    # -----------------------
    def _sample_other_index(self, exclude):
        rand_idx = random.randint(0, self.num_samples - 1)
        while rand_idx == exclude:
            rand_idx = random.randint(0, self.num_samples - 1)
        return rand_idx

    def _neg_A_replace_modality(self, index, modality):
        """
        Cross-sample replacement for one modality only.
        """
        rand_idx = self._sample_other_index(exclude=index)
        if modality == 'T':
            return self.text[rand_idx].clone()
        elif modality == 'A':
            return self.audio[rand_idx].clone()
        else:
            return self.vision[rand_idx].clone()

    def _neg_B_shift(self, x):
        """
        Time shift; if length <= 1, shifting doesn't change, caller should fallback.
        """
        return self.shift_sequence(x)

    def _neg_C_noise_text(self, text):
        """
        Replace one token embedding with noise.
        """
        out = text.clone()
        if out.size(0) > 0:
            idx_word = random.randint(0, out.size(0) - 1)
            out[idx_word] = torch.randn_like(out[idx_word]) * self.noise_std
        return out

    def _neg_C_noise_av(self, x):
        """
        Add Gaussian noise to audio/vision.
        """
        return x + torch.randn_like(x) * self.noise_std

    def _gen_one_modality_neg(self, index, pos_tensor, modality):
        """
        Generate negative for a single modality, ensuring it is meaningfully different from pos.
        Strategies:
          A: replace from other sample (recommended and always valid)
          B: shift (valid when len>1)
          C: noise (always valid)
        """
        strat = random.choices(['A', 'B', 'C'], weights=self.neg_strategy_weights, k=1)[0]

        if strat == 'A':
            return self._neg_A_replace_modality(index, modality)

        if strat == 'B':
            out = self._neg_B_shift(pos_tensor)
            # if length<=1 shift yields same tensor -> fallback to A
            if out.size(0) <= 1:
                return self._neg_A_replace_modality(index, modality)
            return out

        # strat == 'C'
        if modality == 'T':
            return self._neg_C_noise_text(pos_tensor)
        else:
            return self._neg_C_noise_av(pos_tensor)

    def generate_negative_sample(self, index, text_pos, audio_pos, vision_pos):
        """
        Generate (text_neg, audio_neg, vision_neg) for correlation pretraining.

        IMPORTANT FIX:
        - neg_mode='independent' (default): generate each modality neg independently
          => guarantees your 6 triplet terms won't suffer neg==pos degeneration.
        - neg_mode='original': keep old behavior (for ablation / reproduction).
        """
        if self.neg_mode == 'original':
            # Your previous behavior (kept for reproducibility)
            text_neg = text_pos.clone()
            audio_neg = audio_pos.clone()
            vision_neg = vision_pos.clone()

            strategies = ['A', 'B', 'C']
            chosen_strategy = random.choice(strategies) if random.random() < 0.75 else 'D'
            if chosen_strategy == 'D':
                chosen_strategy = random.choice(strategies)

            if chosen_strategy == 'A':
                chosen_modality = random.choice(['T', 'A', 'V'])
                rand_idx = self._sample_other_index(exclude=index)
                other_text = self.text[rand_idx]
                other_audio = self.audio[rand_idx]
                other_vision = self.vision[rand_idx]

                if chosen_modality == 'T':
                    text_neg = other_text.clone()
                elif chosen_modality == 'A':
                    audio_neg = other_audio.clone()
                else:
                    vision_neg = other_vision.clone()

            elif chosen_strategy == 'B':
                text_neg = self.shift_sequence(text_neg)
                audio_neg = self.shift_sequence(audio_neg)
                vision_neg = self.shift_sequence(vision_neg)

            elif chosen_strategy == 'C':
                audio_neg = audio_neg + torch.randn_like(audio_neg) * self.noise_std
                vision_neg = vision_neg + torch.randn_like(vision_neg) * self.noise_std
                if text_neg.size(0) > 0:
                    rand_word_idx = random.randint(0, text_neg.size(0) - 1)
                    text_neg[rand_word_idx] = torch.randn_like(text_neg[rand_word_idx]) * self.noise_std

            return text_neg, audio_neg, vision_neg

        # -------- Recommended: independent per-modality negatives --------
        text_neg = self._gen_one_modality_neg(index, text_pos, 'T')
        audio_neg = self._gen_one_modality_neg(index, audio_pos, 'A')
        vision_neg = self._gen_one_modality_neg(index, vision_pos, 'V')
        return text_neg, audio_neg, vision_neg
