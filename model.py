from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import optimizer, Adam

EPSILON = 1e-4
START_LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 200


def masked_mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor, model_num_steps: int = 10):
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2 if model_num_steps != 1 else 1  # normalization of the loss by the
    # model's step number
    return (sq_error * confidence).mean() * num_steps_norm


class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):
    def __init__(
            self,
            tokenizer,
            pose_dims: (int, int) = (137, 2),
            hidden_dim: int = 128,
            text_encoder_depth: int = 2,
            pose_encoder_depth: int = 4,
            encoder_heads: int = 2,
            encoder_dim_feedforward: int = 2048,
            max_seq_size: int = MAX_SEQ_LEN,
            min_seq_size: int = 20,
            num_steps: int = 10,
            tf_p: float = 0.5,
            lr: float = START_LEARNING_RATE,
            noise_epsilon: float = EPSILON,
            seq_len_weight: float = 2e-5,
            optimizer_fn: optimizer = torch.optim.Adam,
            separate_positional_embedding: bool = False,
            num_pose_projection_layers: int = 1,
            concat: bool = True,
            blend: bool = True
    ):
        super().__init__()
        self.lr = lr
        self.noise_epsilon = noise_epsilon
        self.tf_p = tf_p
        self.seq_len_weight = seq_len_weight
        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size
        self.min_seq_size = min_seq_size
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.pose_dims = pose_dims
        self.optimizer_fn = optimizer_fn
        self.separate_positional_embedding = separate_positional_embedding
        self.best_loss = np.inf
        self.concat = concat
        self.blend = blend

        pose_dim = int(np.prod(pose_dims))

        # Embedding layers

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.step_embedding = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=hidden_dim
        )

        if separate_positional_embedding:
            self.pos_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )
            self.text_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )

        else:
            self.positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )

            # positional embedding scalars
            self.alpha_pose = nn.Parameter(torch.randn(1))
            self.alpha_text = nn.Parameter(torch.randn(1))

        if num_pose_projection_layers == 1:
            self.pose_projection = nn.Linear(pose_dim, hidden_dim)
        else:  # Currently only supports 1 or 2 layers
            self.pose_projection = nn.Sequential(
                nn.Linear(pose_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # encoding layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                    dim_feedforward=encoder_dim_feedforward)

        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=text_encoder_depth)
        self.pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=pose_encoder_depth)

        # step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Predict sequence length
        self.seq_length = nn.Linear(hidden_dim, 1)

        # Predict pose difference
        self.pose_diff_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def encode_text(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.text_positional_embeddings(tokenized["positions"])
        else:
            positional_embedding = self.alpha_text * self.positional_embeddings(tokenized["positions"])

        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding
        encoded = self.text_encoder(embedding.transpose(0, 1),
                                    src_key_padding_mask=tokenized["attention_mask"]).transpose(0, 1)

        seq_length = self.seq_length(encoded).mean(axis=1)
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length

    def forward(self, text: str, first_pose: torch.Tensor, sequence_length: int = -1):
        text_encoding, seq_len = self.encode_text([text])
        seq_len = round(float(seq_len))
        seq_len = max(min(seq_len, self.max_seq_size), self.min_seq_size)
        sequence_length = seq_len if sequence_length == -1 else sequence_length
        pose_sequence = {
            "data": first_pose.expand(1, sequence_length, *self.pose_dims),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool, device=self.device),
        }

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            yield pred
        else:
            step_num = 0
            while True:
                yield pose_sequence["data"][0]
                pose_sequence["data"] = self.refinement_step(step_num, pose_sequence, text_encoding)[0]
                step_num += 1

    def refinement_step(self, step_num, pose_sequence, text_encoding):
        batch_size = pose_sequence["data"].shape[0]
        pose_sequence["data"] = pose_sequence["data"].detach()  # Detach from graph
        batch_step_num = torch.repeat_interleave(torch.LongTensor([step_num]),
                                                 batch_size).unsqueeze(1).to(self.device)
        step_encoding = self.step_encoder(self.step_embedding(batch_step_num))
        change_pred = self.refine_pose_sequence(pose_sequence, text_encoding, step_encoding)
        cur_step_size = self.get_step_size(step_num+1)
        prev_step_size = self.get_step_size(step_num) if step_num > 0 else 0
        step_size = cur_step_size-prev_step_size
        if self.blend:
            pred = (1-step_size) * pose_sequence["data"] + step_size * change_pred
        else:
            pred = pose_sequence["data"] + step_size * change_pred  # add
        return pred, cur_step_size

    def embed_pose(self, pose_sequence_data):
        batch_size, seq_length, _, _ = pose_sequence_data.shape
        flat_pose_data = pose_sequence_data.reshape(batch_size, seq_length, -1)

        positions = torch.arange(0, seq_length, dtype=torch.long, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.pos_positional_embeddings(positions)
        else:
            positional_embedding = self.alpha_pose * self.positional_embeddings(positions)

        # Encode pose sequence
        pose_embedding = self.pose_projection(flat_pose_data) + positional_embedding
        return pose_embedding

    def encode_pose(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape

        # Encode pose sequence
        pose_embedding = self.embed_pose(pose_sequence["data"])

        if step_encoding is not None:
            step_mask = torch.zeros([step_encoding.size(0), 1], dtype=torch.bool, device=self.device)

        pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"], step_encoding], dim=1)
        pose_text_mask = torch.cat(
            [pose_sequence["mask"], text_encoding["mask"], step_mask], dim=1
        )

        pose_encoding = self.__get_text_pose_encoder()(
            pose_text_sequence.transpose(0, 1), src_key_padding_mask=pose_text_mask
        ).transpose(0, 1)[:, :seq_length, :]

        return pose_encoding

    def __get_text_pose_encoder(self):
        if hasattr(self, "text_pose_encoder"):
            return self.text_pose_encoder
        else:
            return self.pose_encoder

    def refine_pose_sequence(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_encoding = self.encode_pose(pose_sequence, text_encoding, step_encoding)

        # Predict desired change
        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_dims)

    def get_step_size(self, step_num):
        if step_num < 2:
            return 0.1
        else:
            return np.log(step_num) / np.log(self.num_steps)

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="validation")

    def step(self, batch, *unused_args, phase: str):
        """
        @param batch: data batch
        @param phase: either "train" or "validation"
        """
        text_encoding, sequence_length = self.encode_text(batch["text"])
        pose = batch["pose"]

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, num_keypoints, _ = pose["data"].shape

        pose_sequence = {
            "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"])
        }

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            l1_gold = pose["data"]
            refinement_loss = masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
        else:
            refinement_loss = 0
            for i in range(self.num_steps):
                pred, step_size = self.refinement_step(i, pose_sequence, text_encoding)
                l1_gold = step_size * pose["data"] + (1 - step_size) * pose_sequence["data"]
                refinement_loss += masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)

                teacher_forcing_step_level = np.random.rand(1)[0] < self.tf_p
                pose_sequence["data"] = l1_gold if phase == "validation" or teacher_forcing_step_level else pred

                if phase == "train":  # add just a little noise while training
                    pose_sequence["data"] = pose_sequence["data"] + torch.randn_like(pose_sequence["data"]) * \
                                            self.noise_epsilon

        sequence_length_loss = F.mse_loss(sequence_length, pose["length"])
        loss = refinement_loss + self.seq_len_weight * sequence_length_loss

        self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
