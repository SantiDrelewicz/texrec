from .model import Model
from .utils.dataset import PunctCapitalDataset, collate_fn
from .utils.dataloader import create_dataloader

import os
from typing import Callable, Optional

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from transformers import BertTokenizer, BertTokenizerFast


class TexRec:
    def __init__(
        self,
        hidden_dim: int = 128,
        lstm: bool = False,
        bidirectional: bool = False,
        n_layers: int = 1,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = lstm
        self.n_layers = n_layers
        self.dropout = dropout

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        self.tokenizer_fast = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-cased")

        self.model = Model(hidden_size=hidden_dim,
                           bidirectional=bidirectional,
                           lstm=lstm,
                           num_layers=n_layers,
                           dropout=dropout).to(self.device)

        self.batch_size = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = None
        self.optimizer = None
        self.lr_scheduler_patience = None
        self.early_stopping_patience = None
        self.epochs = None

        self.is_fitted = False

        self.idx_map_init = {0: "", 1: "¿"}
        self.idx_map_final = {0: "", 1: ".", 2: "?", 3: ","}


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
        init_loss = self.criterion(logits["init_punct"].view(-1, 2), target["init_punct"].view(-1))
        final_loss = self.criterion(logits["final_punct"].view(-1, 4), target["final_punct"].view(-1))
        cap_loss = self.criterion(logits["capital"].view(-1, 4), target["capital"].view(-1))
        return init_loss + final_loss + cap_loss


    @staticmethod
    def _classif_report(all_trues: dict[str, list],
                        all_preds: dict[str, list],
                        epoch: int,
                        output_dict: bool,
                        mode: str) -> None | dict[str, dict]:
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


    def _train_step(self, train_loader: DataLoader,
                    epoch: int, output_dict: bool = False) -> float | dict[str, float | dict]:
        """Entrena una época del modelo y devuelve la pérdida en la época"""
        all_trues: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        all_preds: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        self.model.train()
        training_loss = 0.0
        for input_ids, target in train_loader:
            self.optimizer.zero_grad()
            with autocast("cuda"):
              logits = self.model(input_ids)
            loss = self._loss_fn(logits, target)
            loss.backward()
            self.optimizer.step()
            training_loss += loss.item()

            preds = TexRec._calc_preds_from_logits(logits)
            TexRec._extend_all_preds_and_trues(
                all_preds, all_trues,
                preds, target
            )
        classif_report = TexRec._classif_report(all_trues, all_preds,
                                                             output_dict=output_dict,
                                                             epoch=epoch, mode="train")
        avg_train_loss = training_loss / len(train_loader)

        output = {"loss": avg_train_loss}
        if classif_report is not None:
            output["classif_report"] = classif_report
        return output


    def _eval_step(self, val_loader: DataLoader,
                   epoch: int, output_dict: bool = False) -> float | dict[str, float | dict]:
        """Evalúa el modelo en el conjunto de validación y devuelve la pérdida"""
        all_trues: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        all_preds: dict[str, list] = {
            "init_punct": [], "final_punct": [], "capital": []
        }
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, target in val_loader:
                logits = self.model(input_ids)
                loss = self._loss_fn(logits, target)
                val_loss += loss.item()
                preds = TexRec._calc_preds_from_logits(logits)
                TexRec._extend_all_preds_and_trues(
                    all_preds, all_trues,
                    preds, target
                )
        classif_report = TexRec._classif_report(all_trues, all_preds,
                                                             epoch=epoch, mode="val",
                                                             output_dict=output_dict)
        avg_val_loss = val_loss / len(val_loader)

        output = {"loss": avg_val_loss}
        if classif_report is not None:
            output["classif_report"] = classif_report
        return output


    def train(
        self,
        train_data: list[str], val_data: list[str],
        epochs: int = 1, batch_size: int = 1,
        optimizer: torch.optim = torch.optim.SGD, lr: float = 1e-3,
        lr_scheduler_patience: int = 2,
        early_stopping_patience: int = 3,
        output_classif_report_dict: bool = False
    ):
        self.batch_size = batch_size
        self.learning_rate = lr
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.epochs = epochs

        print(f"Extracting labels from data")
        train_loader, val_loader = create_dataloader(train_data, self.tokenizer,                                                        batch_size, collate_fn)

        losses: dict[str, list[float]] = {"train": [], "val": []}

        if output_classif_report_dict:
          classif_reports = {"train": {"init_punct": [], "final_punct": [], "capital": []},
                             "val"  : {"init_punct": [], "final_punct": [], "capital": []}}

        scaler = GradScaler("cuda")

        print("Training and validating model...")
        for epoch in range(1, epochs+1):
            train_metrics = self._train_step(train_loader, epoch, output_dict=output_classif_report_dict)
            train_loss = train_metrics["loss"]
            losses["train"].append(train_loss)
            val_metrics = self._eval_step(val_loader, epoch, output_dict=output_classif_report_dict)
            val_loss = val_metrics["loss"]
            losses["val"].append(val_metrics["loss"])

            if output_classif_report_dict:
                train_cr, val_cr = train_metrics["classif_report"], val_metrics["classif_report"]

                classif_reports["train"]["init_punct"].append(train_cr["init_punct"])
                classif_reports["train"]["final_punct"].append(train_cr["final_punct"])
                classif_reports["train"]["capital"].append(train_cr["capital"])

                classif_reports["val"]["init_punct"].append(val_cr["init_punct"])
                classif_reports["val"]["final_punct"].append(val_cr["final_punct"])
                classif_reports["val"]["capital"].append(val_cr["capital"])

            print(f"Epoch {epoch}/{epochs}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        self.is_fitted = True

        TexRec.plot_loss_curves(losses)
        print("Training and validation completed")

        output = {"losses": losses}
        if output_classif_report_dict:
          return {"losses": losses, "classif_reports": classif_reports}
        return output


    @staticmethod
    def plot_loss_curves(losses: dict[str, list]):
        """Plot training and validation losses"""
        plt.plot(losses["train"], label="train")
        plt.plot(losses["val"], label="val")
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
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids)

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
            self.model.eval()
            with torch.no_grad():
                init_logits, final_logits, cap_logits = self.model(input_ids_tensor)

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
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "lstm": self.lstm,
            },
            "training_config": {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            },
            "is_fitted": self.is_fitted,
        }

        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")


    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = torch.load(filepath, map_location=self.device)

        # Update configs
        config = model_data["model_config"]
        self.hidden_dim = config["hidden_dim"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]
        self.lstm = config["lstm"]
        self.bidirectional = config["bidirectional"]

        train_config = model_data["training_config"]
        self.learning_rate = train_config["learning_rate"]
        self.batch_size = train_config["batch_size"]

        # Recreate model
        self.model = RNN(
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            lstm=self.lstm
        ).to(self.device)


        # Load state
        self.model.load_state_dict(model_data["model_state_dict"])
        self.is_fitted = model_data["is_fitted"]

        print(f"Model loaded from {filepath}")