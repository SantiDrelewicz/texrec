from ..engine.model import Model
from ..utils.dataset import collate_fn
from ..utils.dataloader import create_dataloader
from ..utils.constants import TOKENIZER_EMBEDDING_MODEL_NAME

import os
from contextlib import nullcontext

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import torch
import torch.optim.lr_scheduler
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import BertTokenizer, BertTokenizerFast


class TexRecModel():
    def __init__(
        self,
        hidden_dim: int = 128,
        bidirectional: bool = False,
        n_layers: int = 1,
        dropout: float = 0.1,
        lstm: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        # model params
        self._hidden_dim = hidden_dim
        self._bidirectional = bidirectional
        self._lstm = lstm
        self._n_layers = n_layers
        self._dropout = dropout

        self.device = device

        self._model = Model(hidden_size=hidden_dim,
                            bidirectional=bidirectional,
                            lstm=lstm,
                            num_layers=n_layers,
                            dropout=dropout).to(self.device)

        # tokenizers
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_EMBEDDING_MODEL_NAME)
        self.tokenizer_fast = BertTokenizerFast.from_pretrained(TOKENIZER_EMBEDDING_MODEL_NAME)

        self._criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.metrics = {}

        self.is_fitted = False

        self.idx_map_init = {0: "", 1: "¿"}
        self.idx_map_final = {0: "", 1: ".", 2: "?", 3: ","}


    @property
    def n_params(self) -> int:
        """Number of trainable parameters in the model"""
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)


    @staticmethod
    def _calc_preds_from_logits(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Calculate predictions from logits"""
        preds = {}
        preds["init_punct"] = logits["init_punct"].argmax(dim=-1)
        preds["final_punct"] = logits["final_punct"].argmax(dim=-1)
        preds["capital"] = logits["capital"].argmax(dim=-1)
        return preds


    @staticmethod
    def _extend_all_preds_and_trues(
        all_preds: dict[str, list], all_trues: dict[str, list],
        preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ):
      mask_init = target["init_punct"].view(-1) != -100
      mask_final = target["final_punct"].view(-1) != -100
      mask_cap = target["capital"].view(-1) != -100

      all_preds["init_punct"].extend(preds["init_punct"].view(-1)[mask_init].cpu().tolist())
      all_preds["final_punct"].extend(preds["final_punct"].view(-1)[mask_final].cpu().tolist())
      all_preds["capital"].extend(preds["capital"].view(-1)[mask_cap].cpu().tolist())

      all_trues["init_punct"].extend(target["init_punct"].view(-1)[mask_init].cpu().tolist())
      all_trues["final_punct"].extend(target["final_punct"].view(-1)[mask_final].cpu().tolist())
      all_trues["capital"].extend(target["capital"].view(-1)[mask_cap].cpu().tolist())


    def _loss_fn(self, logits: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> float:
        """Calculate the total loss for a batch of data"""
        init_loss = self._criterion(logits["init_punct"].view(-1, 2), target["init_punct"].view(-1))
        final_loss = self._criterion(logits["final_punct"].view(-1, 4), target["final_punct"].view(-1))
        cap_loss = self._criterion(logits["capital"].view(-1, 4), target["capital"].view(-1))
        return init_loss + final_loss + cap_loss


    @staticmethod
    def _classif_report(all_trues: dict[str, list],
                        all_preds: dict[str, list],
                        epoch: int,
                        mode: str,
                        output_dict: bool = True) -> None | dict[str, dict]:
        cr = {}
        cr["init_punct"] = classification_report(
            all_trues["init_punct"], all_preds["init_punct"],
            labels=[0, 1], target_names=['none', '¿'],
            zero_division=0, output_dict=output_dict
        )
        cr["final_punct"] = classification_report(
            all_trues["final_punct"], all_preds["final_punct"],
            labels=[0, 1, 2, 3], target_names=['space', ',', '.', '?'],
            zero_division=0, output_dict=output_dict
        )
        cr["capital"] = classification_report(
            all_trues["capital"], all_preds["capital"],
            labels=[0, 1, 2, 3], target_names=["Lower", "Initial", "Mixed", "ALLCAP"],
            zero_division=0, output_dict=output_dict
        )
        if output_dict:
          return cr
        else:
          text = []
          text.append(f"Epoch {epoch}:\n")
          text.append("Puntuación inicial por clase:\n")
          text.append(cr["init_punct"])
          text.append("\nPuntuación final por clase:\n")
          text.append(cr["final_punct"])
          text.append("\nCapitalización por clase:\n")
          text.append(cr["capital"])
          full_text = "\n".join(text)

          # Chequear si ya existe el classif_report.txt
          if not os.path.exists(f"{mode}_classif_report.txt"):
            with open(f"{mode}_classif_report.txt", "w", encoding="utf-8") as f:
              f.write(full_text + "\n\n")
          else:
            with open(f"{mode}_classif_report.txt", "a", encoding="utf-8") as f:
              f.write(full_text + "\n\n")


    def _train_epoch(self, train_loader: DataLoader,
                    epoch: int, output_dict: bool = True) -> float | dict[str, float | dict]:
        """Entrena una época del modelo y devuelve la pérdida en la época"""
        all_trues: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        all_preds: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        self._model.train()
        training_loss = 0.0
        for input_ids, target in train_loader:
            self._optimizer.zero_grad()

            with (
                nullcontext()
                if self.device.type == "cpu" and self._lstm
                else autocast(self.device.type)
            ):
                logits = self._model(input_ids)
                loss = self._loss_fn(logits, target)

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._scaler.step(self._optimizer)
            self._scaler.update()

            training_loss += loss.item()

            preds = TexRecModel._calc_preds_from_logits(logits)
            TexRecModel._extend_all_preds_and_trues(
                all_preds, all_trues,
                preds, target
            )

        classif_report = TexRecModel._classif_report(
            all_trues, all_preds,
            output_dict=output_dict,
            epoch=epoch,
            mode="train"
        )
        avg_train_loss = training_loss / len(train_loader)

        output = {"loss": avg_train_loss}
        if classif_report is not None:
            output["classif_report"] = classif_report
        return output


    def _val_epoch(self, val_loader: DataLoader,
                   epoch: int, output_dict: bool = True) -> float | dict[str, float | dict]:
        """Evalúa el modelo en el conjunto de validación y devuelve la pérdida"""
        all_trues: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        all_preds: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        self._model.eval()
        val_loss = 0.0
        with torch.no_grad():

            for input_ids, target in val_loader:

                with (
                    nullcontext()
                    if self.device.type == "cpu" and self._lstm
                    else autocast(self.device.type)
                ):
                    logits = self._model(input_ids)
                    loss = self._loss_fn(logits, target)

                val_loss += loss.item()
                preds = TexRecModel._calc_preds_from_logits(logits)
                TexRecModel._extend_all_preds_and_trues(
                    all_preds, all_trues,
                    preds, target
                )

        classif_report = TexRecModel._classif_report(
            all_trues, all_preds,
            epoch=epoch,
            mode="val",
            output_dict=output_dict
        )
        avg_val_loss = val_loss / len(val_loader)

        output = {"loss": avg_val_loss}
        if classif_report is not None:
            output["classif_report"] = classif_report
        return output


    def train(
        self,
        train_data: list[str], val_data: list[str],
        batch_size: int = 1,
        num_workers: int = 0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optimizer: torch.optim = torch.optim.SGD,
        lr: float = 1e-3,
        lr_scheduler_factor: float = 0.1,
        lr_scheduler_patience: int = 0,
        early_stopping_patience: int = 2,
        output_classif_report_dict: bool = True,
        plot_losses: bool = False
    ):
        self._batch_size = batch_size
        self._learning_rate = lr
        self._optimizer = optimizer(self._model.parameters(), lr=self._learning_rate)
        self._lr_scheduler_patience = lr_scheduler_patience
        self._early_stopping_patience = early_stopping_patience

        print(f"Extracting labels from data")
        train_loader = create_dataloader(
            train_data, self.tokenizer, self._batch_size, num_workers=num_workers, device=device
        )
        val_loader = create_dataloader(
            val_data, self.tokenizer, self._batch_size, num_workers=num_workers, device=device
        )

        self.metrics = {
            "losses": {"train": [], "val": []}
        }

        if output_classif_report_dict:
            self.metrics["classif_reports"] = {
                "train": {"init_punct": [], "final_punct": [], "capital": []},
                "val": {"init_punct": [], "final_punct": [], "capital": []}
            }

        patience_counter = 0
        self._scaler = GradScaler(self.device.type)
        self._epochs = 1
        best_avg_val_loss = float("inf")

        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            verbose=True
        )

        # Set up auxiliar metrics in case scheduler activates
        aux_metrics = {"losses": {"train": [], "val": []}}
        if output_classif_report_dict:
            aux_metrics["classif_reports"] = {
                "train": {"init_punct": [], "final_punct": [], "capital": []},
                "val": {"init_punct": [], "final_punct": [], "capital": []}
            }

        print(f"Training model with {self.n_params} parameters on {self.device.type}")
        while True:
            self.is_fitted = True

            train_metrics = self._train_epoch(train_loader, self._epochs, output_dict=output_classif_report_dict)
            avg_train_loss = train_metrics["loss"]
            val_metrics = self._val_epoch(val_loader, self._epochs, output_dict=output_classif_report_dict)
            avg_val_loss = val_metrics["loss"]

            # Update aux_metrics
            aux_metrics["losses"]["train"].append(avg_train_loss)
            aux_metrics["losses"]["val"].append(avg_val_loss)

            if output_classif_report_dict:
                train_cr, val_cr = train_metrics["classif_report"], val_metrics["classif_report"]

                aux_metrics["classif_reports"]["train"]["init_punct"].append(train_cr["init_punct"])
                aux_metrics["classif_reports"]["train"]["final_punct"].append(train_cr["final_punct"])
                aux_metrics["classif_reports"]["train"]["capital"].append(train_cr["capital"])

                aux_metrics["classif_reports"]["val"]["init_punct"].append(val_cr["init_punct"])
                aux_metrics["classif_reports"]["val"]["final_punct"].append(val_cr["final_punct"])
                aux_metrics["classif_reports"]["val"]["capital"].append(val_cr["capital"])

            self._scheduler.step(avg_val_loss)
            if avg_val_loss < best_avg_val_loss:
                best_avg_val_loss = avg_val_loss
                patience_counter = 0

                # Update metrics
                self.metrics["losses"]["train"].extend(aux_metrics["losses"]["train"])
                self.metrics["losses"]["val"].extend(aux_metrics["losses"]["val"])
                if output_classif_report_dict:
                    self.metrics["classif_reports"]["train"]["init_punct"].extend(aux_metrics["classif_reports"]["train"]["init_punct"])
                    self.metrics["classif_reports"]["train"]["final_punct"].extend(aux_metrics["classif_reports"]["train"]["final_punct"])
                    self.metrics["classif_reports"]["train"]["capital"].extend(aux_metrics["classif_reports"]["train"]["capital"])

                    self.metrics["classif_reports"]["val"]["init_punct"].extend(aux_metrics["classif_reports"]["val"]["init_punct"])
                    self.metrics["classif_reports"]["val"]["final_punct"].extend(aux_metrics["classif_reports"]["val"]["final_punct"])
                    self.metrics["classif_reports"]["val"]["capital"].extend(aux_metrics["classif_reports"]["val"]["capital"])

                self.save_model("texrec_model.pt")

                # Reset aux_metrics
                aux_metrics = {"losses": {"train": [], "val": []}}
                if output_classif_report_dict:
                    aux_metrics["classif_reports"] = {
                        "train": {"init_punct": [], "final_punct": [], "capital": []},
                        "val": {"init_punct": [], "final_punct": [], "capital": []}
                    }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {self._epochs}.")
                    break

            self._learning_rate = self._optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {self._epochs}: Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}, patience: {patience_counter}, lr: {self._learning_rate:.6f}"
            )
            self._epochs += 1

        if plot_losses:
            self.plot_loss_curves(self.metrics["losses"])

        print("Training completed")


    def plot_loss_curves(self):
        """Plot training and validation losses"""
        plt.plot(self.metrics["losses"]["train"], label="train")
        plt.plot(self.metrics["losses"]["val"], label="val")
        plt.legend()
        plt.show()


    def predict(self, text: str):
        """Predict punctuation and capitalization for input raw text"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        return self._predict_and_reconstruct(text)


    def _predict_and_reconstruct(self, raw_sentence: str):
        """
        Runs the model on a single raw sentence and returns the
        reconstructed sentence with punctuation & capitalization.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        # Fix: Removed the extra call to tokenizer_fast
        enc = self.tokenizer_fast(
            raw_sentence.lower().split(),  # split into words so word_ids works
            is_split_into_words=True,
            return_offsets_mapping=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = enc["input_ids"].to(self.device)
        word_ids = enc.word_ids(batch_index=0)  # list of length L

        # 2) Model forward
        self._model.eval()
        with torch.no_grad():
            logits = self._model(input_ids)

        init_pred = logits["init_punct"].argmax(dim=-1).squeeze(0).cpu().tolist()
        final_pred = logits["final_punct"].argmax(dim=-1).squeeze(0).cpu().tolist()
        cap_pred = logits["capital"].argmax(dim=-1).squeeze(0).cpu().tolist()

        # 3) Gather per-word predictions
        words: list[str] = []
        cur_word_idx = None
        cur_subtokens: list[str] = []
        cur_init = ""
        cur_cap = 0
        cur_final = ""

        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            token = self.tokenizer_fast.convert_ids_to_tokens(int(input_ids[0, i]))
            # start of a new word?
            if wid != cur_word_idx:
                # flush previous
                if cur_word_idx is not None:
                    # assemble the word text
                    word_text = "".join(cur_subtokens)
                    # apply capitalization
                    if cur_cap == 3:
                        word_text = word_text.upper()
                    elif cur_cap == 1:
                        word_text = word_text.capitalize()
                    elif cur_cap == 2:
                        if len(word_text) > 1:
                            word_text = word_text[0].upper() + word_text[1:]
                        else:
                            word_text = word_text.upper()
                    # attach final punctuation
                    word_text = word_text + self.idx_map_final[cur_final]
                    # prepend initial punctuation if any
                    word_text = cur_init + word_text
                    words.append(word_text)
                # reset for new word
                cur_word_idx = wid
                cur_subtokens = [token.replace("##", "")]  # start fresh
                cur_init = self.idx_map_init[init_pred[i]]
                cur_final = final_pred[i]
                cur_cap = cap_pred[i]
            else:
                # continuing same word
                cur_subtokens.append(token.replace("##", ""))
                # update final & cap to last sub-token's prediction
                cur_final = final_pred[i]
                # we keep init and cap from first subtoken

        # flush last word
        if cur_word_idx is not None:
            word_text = "".join(cur_subtokens)
            if cur_cap == 3:
                word_text = word_text.upper()
            elif cur_cap == 1:
                word_text = word_text.capitalize()
            elif cur_cap == 2:
                word_text = word_text[0].upper() + word_text[1:]
            word_text = word_text + self.idx_map_final[cur_final]
            word_text = cur_init + word_text
            words.append(word_text)

        # finally, join with spaces:
        return " ".join(words)


    def predict_and_fill_csv(self, input_df: pd.DataFrame, output_file: str = "predicted.csv") -> pd.DataFrame:
        """
        Takes a dataframe with columns: instancia_id, token_id, token
        Returns a new dataframe with added columns:
        init_punct, final_punct, capital
        One row per *input token* (same granularity).
        """

        results = []
        tokenizer_fast = self.tokenizer_fast
        device = self.device

        for instancia_id, group in input_df.groupby("instancia_id"):
            # 1. Get the *input tokens exactly as they appear* (these are already subword tokens)
            tokens = group["token"].tolist()

            # 2. Convert tokens to IDs using tokenizer's vocab
            input_ids = tokenizer_fast.convert_tokens_to_ids(tokens)
            input_ids_tensor = torch.tensor([input_ids], device=device)

            # 3. Predict
            self._model.eval()
            with torch.no_grad():
                init_logits, final_logits, cap_logits = self._model(input_ids_tensor)

            # 4. Get predictions per token
            init_pred = init_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            final_pred = final_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            cap_pred = cap_logits.argmax(dim=-1).squeeze(0).cpu().tolist()

            # 5. Decode label indices
            init_punct = [self.idx_map_init[idx] for idx in init_pred]
            final_punct = [self.idx_map_final[idx] for idx in final_pred]
            capital = cap_pred  # leave as integers or map if you want

            # 6. Build output dataframe for this group
            predicted_group = group.copy()
            predicted_group["init_punct"] = init_punct
            predicted_group["final_punct"] = final_punct
            predicted_group["capitalización"] = capital

            results.append(predicted_group)

        # Concatenate all
        final_df = pd.concat(results, ignore_index=True)

        if output_file:
            final_df.to_csv(output_file, index=False)

        return final_df


    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model_state_dict": self._model.state_dict(),
            "model_config": {
                "hidden_dim": self._hidden_dim,
                "n_layers": self._n_layers,
                "dropout": self._dropout,
                "bidirectional": self._bidirectional,
                "lstm": self._lstm,
            },
            "training_config": {
                "final_learning_rate": self._learning_rate,
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "optimizer": self._optimizer.__class__.__name__,
                "scheduler": self._scheduler.__class__.__name__,
                "lr_scheduler_patience": self._lr_scheduler_patience,
                "early_stopping_patience": self._early_stopping_patience,
            },
            "metrics": self.metrics,
        }
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")


    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = torch.load(filepath, map_location=self.device)

        # Update configs
        config = model_data["model_config"]
        self._hidden_dim = config["hidden_dim"]
        self._n_layers = config["n_layers"]
        self._dropout = config["dropout"]
        self._lstm = config["lstm"]
        self._bidirectional = config["bidirectional"]

        self.train_config = model_data["training_config"]
        self.metrics = model_data["metrics"]

        # Recreate model
        self._model = Model(
            hidden_size=self._hidden_dim,
            num_layers=self._n_layers,
            dropout=self._dropout,
            bidirectional=self._bidirectional,
            lstm=self._lstm
        ).to(self.device)

        # Load state
        self._model.load_state_dict(model_data["model_state_dict"])

        self.is_fitted = True

        print(f"Model loaded from {filepath}")