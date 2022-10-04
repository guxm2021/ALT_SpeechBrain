#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with DSing
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_with_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens. Training is performed on the full
DALI dataset.

Authors
 * Xiangming Gu 2022
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import torchaudio
from collections import OrderedDict

logger = logging.getLogger(__name__)

from speechbrain.core import *
from torch.utils.tensorboard import SummaryWriter


# Define training procedure
class ASR(sb.Brain):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        self.writer = SummaryWriter(self.hparams.tensorboard_loc)
        self.step_cnt = 1

    def compute_forward(self, batch, stage, not_use_lm=False):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.emb(tokens_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.VALID:
            if not_use_lm == True:
                hyps = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index
                )
            elif self.hparams.use_lm_in_valid == True:
                self.modules.lm_model.to(self.device)
                hyps, scores = self.hparams.greedy_searcher(x, wav_lens)
            else:
                hyps = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index
                )
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            if self.hparams.use_lm_in_test == True:
                self.modules.lm_model.to(self.device)
                hyps, scores = self.hparams.beam_searcher(x, wav_lens)
            else:
                hyps = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index
                )
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, tokens_eos_lens)
        loss = self.hparams.ctc_weight * loss_ctc
        loss += (1 - self.hparams.ctc_weight) * loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in hyps
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.ctc_metrics.append(ids, p_ctc, tokens, wav_lens, tokens_lens)
            self.seq_metrics.append(ids, p_seq, tokens_eos, tokens_eos_lens)
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss, loss_ctc, loss_seq


    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:
            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    if self._optimizer_step_limit_exceeded:
                        logger.info("Train iteration limit exceeded")
                        break
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

                    # Validation and log by tensorboard, after some steps
                    if self.step % self.hparams.valid_interval == 0:
                        valid_step = 0
                        self.cer_metric = self.hparams.cer_computer()
                        self.wer_metric = self.hparams.error_rate_computer()

                        self.modules.eval()
                        avg_valid_loss = 0.0
                        avg_valid_loss_ctc = 0.0
                        avg_valid_loss_seq = 0.0
                        with torch.no_grad():
                            for batch in tqdm(
                                valid_set, dynamic_ncols=True, disable=not enable
                            ):
                                valid_step += 1
                                loss, loss_ctc, loss_seq = self.evaluate_batch(batch, stage=Stage.VALID, not_use_lm=True)
                                avg_valid_loss -= avg_valid_loss / valid_step
                                avg_valid_loss += float(loss) / valid_step
                                avg_valid_loss_ctc -= avg_valid_loss_ctc / valid_step
                                avg_valid_loss_ctc += float(loss_ctc) / valid_step
                                avg_valid_loss_seq -= avg_valid_loss_seq / valid_step
                                avg_valid_loss_seq += float(loss_seq) / valid_step

                        self.tensorboard_log(avg_valid_loss, avg_valid_loss_ctc, avg_valid_loss_seq)
                        self.checkpointer.save_and_keep_only(
                            meta={"loss": avg_valid_loss, "iterations":self.step_cnt, 'epoch':self.hparams.epoch_counter.current}, 
                            min_keys=["loss"], end_of_epoch=False
                        )
                        self.modules.train()


            # Run train "on_stage_end" on all processes
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss, loss_ctc, loss_seq = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[Stage.VALID, avg_valid_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break


    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            raise NotImplementedError()
            with torch.cuda.amp.autocast():
                predictions = self.compute_forward(batch, sb.Stage.TRAIN)
                loss, loss_ctc, loss_seq = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
            
            self.scaler.scale(loss / self.hparams.accu_steps).backward()

            if self.step % self.hparams.accu_steps == 0:
                self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.adam_optimizer)

                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.adam_optimizer)

                self.scaler.update()
                self.wav2vec_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()
        else:
            predictions = self.compute_forward(batch, sb.Stage.TRAIN)
            loss, loss_ctc, loss_seq = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

            (loss / self.hparams.accu_steps).backward()
            if self.step % self.hparams.accu_steps == 0:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.adam_optimizer.step()

                self.wav2vec_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()

        # Statistics
        self.loss_train_stat['loss'] += loss.detach().item()
        self.loss_ctc_stat['loss'] += loss_ctc.detach().item()
        self.loss_seq_stat['loss'] += loss_seq.detach().item()

        self.step_cnt += 1

        return loss.detach().cpu()

    def tensorboard_log(self, avg_valid_loss, valid_loss_ctc, valid_loss_seq):
        self.writer.add_scalars('1 Loss Curve/1 Overall', {
            'train_loss_total': self.loss_train_stat['loss'] / self.step,
            'validation_loss': avg_valid_loss
        }, self.step_cnt)
        self.writer.add_scalars('1 Loss Curve/2 CTC vs S2S loss', {
            'train_ctc': self.loss_ctc_stat['loss'] / self.step,
            'train_s2s': self.loss_seq_stat['loss'] / self.step,
            'valid_ctc': valid_loss_ctc,
            'valid_s2s': valid_loss_seq,
        }, self.step_cnt)
        self.writer.add_scalars('2 Validation WER Graph', {
            'WER': self.wer_metric.summarize("error_rate"),
            'CER': self.cer_metric.summarize("error_rate")
        }, self.step_cnt)

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            "step": self.step,
            "step_cnt": self.step_cnt,
            "avg_train_loss": self.avg_train_loss,
            "optimizer_step": self.optimizer_step,
        }
        with open(path, "w") as w:
            w.write(yaml.dump(save_dict))

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch, device):
        del end_of_epoch
        del device
        with open(path) as f:
            save_dict = yaml.safe_load(f)
        self.step = save_dict["step"]
        self.avg_train_loss = save_dict["avg_train_loss"]
        self.optimizer_step = save_dict["optimizer_step"]
        if 'step_cnt' in save_dict:
            self.step_cnt = save_dict['step_cnt']

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                loss, _, _ = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)
                
                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            )
        self.step = 0
        return avg_test_loss

    def evaluate_batch(self, batch, stage, not_use_lm=False):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage, not_use_lm=not_use_lm)
        loss, loss_ctc, loss_seq = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach(), loss_ctc.detach(), loss_seq.detach()
    
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )
        
        if self.hparams.pretrain:
            logger.info(f"Pretrain wav2vec 2.0 and encoder-decoder model from the folder: {self.hparams.pretrain_folder}")
            self.modules.wav2vec2.load_state_dict(torch.load(os.path.join(self.hparams.pretrain_folder, "wav2vec2.pt")))
            self.hparams.model.load_state_dict(torch.load(os.path.join(self.hparams.pretrain_folder, "model.pt")))
        else:
            logger.info("No pretrained wav2vec 2.0 and encoder-decoder model")
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """
        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )
        # save wav2vec 2.0 and encoder-decoder model
        if self.hparams.save_model:
            os.makedirs(self.hparams.save_model_folder, exist_ok=True)
            torch.save(self.modules.wav2vec2.state_dict(), os.path.join(self.hparams.save_model_folder, 'wav2vec2.pt'))
            torch.save(self.hparams.model.state_dict(), os.path.join(self.hparams.save_model_folder, 'model.pt'))
            logger.info(f"Save wav2vec 2.0 and encoder-decoder model to the folder: {self.hparams.save_model_folder}")
        else:
            logger.info("No wav2vec 2.0 and encoder-decoder model to save")
        self.step = 0

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()

        self.loss_train_stat = {'loss': 0.0, 'batch': 0}
        self.loss_ctc_stat = {'loss': 0.0, 'batch': 0}
        self.loss_seq_stat = {'loss': 0.0, 'batch': 0}

        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_adam(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "CER": stage_stats["CER"],
                    "WER": stage_stats["WER"],
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        audio, fs = torchaudio.load(wav)
        assert fs == 16000
        if audio.shape[0] == 1:
            sig = audio.transpose(0, 1).squeeze(1)
        elif audio.shape[0] == 2:
            sig = audio.mean(dim=0)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = hparams["label_encoder"] #os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "char_list", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets, label_encoder


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(
        hparams
    )

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], 
            min_key="loss",        # changed by Xiangming, use min_key "loss" to load checkpointer
            test_loader_kwargs=hparams["test_dataloader_opts"]
        )