import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import torchaudio
import os
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from datasets.movie_dataset import BaseAudioDataset
from losses.stft_loss import MultiResolutionSTFTLoss
from losses.audio_loss import AmplitudeLoss, SDRloss, Envelop_distance
from losses.passt import PaSST
from losses.IB_score import IB_score_metric
from losses.temporal_alignment import WassersteinDistance
from models.hdemucs import HDemucsRemixer
from models.attention import AV_Transformer
from pytorch_lightning import seed_everything

def load_model(config):
    model_config = config['model']
    model_class = globals()[model_config['name']]
    return model_class(**model_config['params'])


class VisAH(pl.LightningModule):
    def __init__(self, model, config, output_dir="./outputs"):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = model
        self.output_dir = output_dir
        self.config = config
        
        self.mr_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[2048, 1024, 512], hop_sizes=[1024, 512, 256], win_lengths=[2048, 1024, 512]
        )
        self.amp_loss = AmplitudeLoss(**config["metrics"]["amplitude"])
        self.sdr_loss = SDRloss()

        # Load checkpoint if specified in config
        checkpoint_path = self.config.get("testing", {}).get("checkpoint_path")
        if checkpoint_path:
            self.load_model_checkpoint(checkpoint_path)

        if self.config["mode"] == "test":
            self.envelope_loss = Envelop_distance()
            self.amp_error = AmplitudeLoss(**config["metrics"]["amplitude"])
            self.ib_score = IB_score_metric(device=self.device)
            self.wasserstein_distance = WassersteinDistance()
            self.passt = PaSST(device=self.device)
            self.runtime = []

    def load_model_checkpoint(self, checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Remove dac_model related keys from the checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint  # Some checkpoints might directly be state_dicts
        
        # Load the cleaned state dict into the model
        self.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully.")

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        # validation_step defines the validation loop. It is independent of forward
        input_audio, target_audio, frame_feats = batch["input_audio"], batch["target_audio"], batch["frame_feats"]
        text_feats = batch["text_feats"]

        pred_audio, gt_mask, pred_mask, weight = self.model(input_audio, target_audio, frame_feats)
        loss = self.mr_stft_loss(pred_audio.squeeze(1), target_audio.squeeze(1))[1]

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["data"]["validation"]["batch_size"])

    def test_step(self, batch, batch_idx):
        self.model.eval()
        # test_step defines the test loop. It is independent of forward
        input_audio, target_audio, frame_feats = batch["input_audio"], batch["target_audio"], batch["frame_feats"]
        text_feats = batch["text_feats"]
        text_caption = batch["text"]

        pred_audio, gt_mask, pred_mask, weight = self.model(input_audio, target_audio, frame_feats)
        loss = self.mr_stft_loss(pred_audio.squeeze(1), target_audio.squeeze(1))[1]

        # self.save_output(input_audio, pred_audio, target_audio, batch["folder_path"])

        # Calculate metrics
        amp_error_pred, amp_error_input, env_dist_pred, env_dist_input, wasserstein_dist_input, wasserstein_dist_pred, ib_score, kl_it_passt, kl_pt_passt = self.calculate_metrics(input_audio, pred_audio, target_audio, batch["folder_path"], text_caption)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("amp_error_pred", amp_error_pred, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("amp_error_input", amp_error_input, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("env_dist_pred", env_dist_pred, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("env_dist_input", env_dist_input, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("wasserstein_dist_input", wasserstein_dist_input, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("wasserstein_dist_pred", wasserstein_dist_pred, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("ib_score_input", ib_score[0].item(), prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("ib_score_pred", ib_score[1].item(), prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("ib_score_target", ib_score[2].item(), prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])        
        self.log("kl_it_passt", kl_it_passt, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])
        self.log("kl_pt_passt", kl_pt_passt, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.config["testing"]["batch_size"])

    def calculate_metrics(self, input_audio, pred_audio, target_audio, folder_path, text_caption=None):
        kl_it_passt = []
        kl_pt_passt = []
        ib_score = []
        for i in range(pred_audio.shape[0]):
            folder_name = os.path.basename(folder_path[i])

            # calculate IB_score
            ib_score.append(self.ib_score(self.output_dir, folder_path[i], self.device))

            folde_path_ = os.path.join(self.output_dir, folder_name)
            file_paths = os.listdir(folde_path_)
            file_paths = [os.path.join(folde_path_, file_path) for file_path in file_paths]

            # caculate passt logits
            passt_logits = self.passt(file_paths, self.device)
            kl_softmax_passt_input_target = torch.nn.functional.kl_div(
                (passt_logits[0:1].softmax(dim=1) + 1e-6).log(),
                passt_logits[2:3].softmax(dim=1),
                reduction="sum",
            ) / len(passt_logits[0:1])
            kl_softmax_passt_pred_target = torch.nn.functional.kl_div(
                (passt_logits[1:2].softmax(dim=1) + 1e-6).log(),
                passt_logits[2:3].softmax(dim=1),
                reduction="sum",
            ) / len(passt_logits[1:2])
            kl_it_passt.append(kl_softmax_passt_input_target)
            kl_pt_passt.append(kl_softmax_passt_pred_target) 

        kl_it_passt = torch.stack(kl_it_passt).mean()
        kl_pt_passt = torch.stack(kl_pt_passt).mean()
        ib_score = torch.stack(ib_score).mean(dim=0)
        amp_error_pred = self.amp_error(pred_audio, target_audio)
        amp_error_input = self.amp_error(input_audio, target_audio)
        env_dist_pred = self.envelope_loss(pred_audio, target_audio)
        env_dist_input = self.envelope_loss(input_audio, target_audio)
        wasserstein_dist_pred = self.wasserstein_distance(target_audio, pred_audio).to(self.device)
        wasserstein_dist_input = self.wasserstein_distance(target_audio, input_audio).to(self.device)
        
        return amp_error_pred, amp_error_input, env_dist_pred, env_dist_input, wasserstein_dist_input, wasserstein_dist_pred, ib_score, kl_it_passt, kl_pt_passt

    def forward(self, x):
        return self.model(x)
    
    def save_output(self, input_audio, pred_audio, target_audio, folder_path, sample_rate=44100):
        for i in range(pred_audio.shape[0]):
            folder_name = os.path.basename(folder_path[i])
            os.makedirs(os.path.join(self.output_dir, folder_name), exist_ok=True)
            input_audio_i = input_audio[i].detach().cpu()
            pred_audio_i = pred_audio[i].detach().cpu()
            target_audio_i = target_audio[i].detach().cpu()
            torchaudio.save(
                os.path.join(self.output_dir, folder_name, f"input_audio.wav"),
                input_audio_i,
                sample_rate,
            )
            torchaudio.save(
                os.path.join(self.output_dir, folder_name, f"pred_audio.wav"),
                pred_audio_i,
                sample_rate,
            )
            torchaudio.save(
                os.path.join(self.output_dir, folder_name, f"target_audio.wav"),
                target_audio_i,
                sample_rate,
            )

    def training_step(self, batch, batch_idx):
        self.model.train()
        # training_step defines the train loop. It is independent of forward
        input_audio, target_audio, frame_feats = batch["input_audio"], batch["target_audio"], batch["frame_feats"]
        text_feats = batch["text_feats"]

        pred_audio, gt_mask, pred_mask, weight = self.model(input_audio, target_audio, frame_feats)
        loss = self.mr_stft_loss(pred_audio.squeeze(1), target_audio.squeeze(1))[1]
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.config["data"]["training"]["batch_size"])
        return loss

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.config.get("training", {}).get("learning_rate", 1e-4))
        return optimizer

def train(config):

    # -------------------
    # Step 2: Define data
    # -------------------
    train_dataset = BaseAudioDataset(mode="train", num_samples=config["data"]["training"]["num_samples"], num_time_frames=config["data"]["num_time_frames"])
    val_dataset = BaseAudioDataset(mode="val", num_samples=config["data"]["validation"]["num_samples"], num_time_frames=config["data"]["num_time_frames"])
    train_loader = data.DataLoader(train_dataset, batch_size=config["data"]["training"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])
    val_loader = data.DataLoader(val_dataset, batch_size=config["data"]["validation"]["batch_size"], num_workers=config["data"]["num_workers"])

    # -------------------
    # Step 3: Define model
    # -------------------
    # transformer
    latent_transformer = AV_Transformer(config["model"]["params"]["in_channels"], config["model"]["params"]["num_heads"], config["model"]["params"]["d_head"], depth=config["model"]["params"]["depth"], num_layers=config["model"]["params"]["num_context_encoder_layers"],context_dim=config["model"]["params"]["context_dim"])
    rebalancingnet = HDemucsRemixer(sources=["output"], latent_transformer=latent_transformer)

    visah_model = VisAH(rebalancingnet, config=config, output_dir=config["testing"]["output_dir"])

    # Define logger and callbacks
    logger = TensorBoardLogger(config["logging"]["save_dir"], name=config["logging"]["name"])
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpointing"]["dirpath"],
        filename=config["checkpointing"]["filename"],
        save_top_k=config["checkpointing"]["save_top_k"],
        monitor=config["checkpointing"]["monitor"],
        mode=config["checkpointing"]["mode"],
    )

    progress_bar = TQDMProgressBar(refresh_rate=config["progress_bar"]["refresh_rate"])  # Update every 10 batches

    trainer = pl.Trainer(        
        max_epochs=config["training"]["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["training"]["devices"],
        strategy=config["training"]["strategy"],
        enable_progress_bar=config["training"]["enable_progress_bar"],)
    if config["training"]["resume_from_checkpoint"]:
        trainer.fit(visah_model, train_loader, val_loader, ckpt_path=config["training"]["resume_from_checkpoint"])
    else:
        trainer.fit(visah_model, train_loader, val_loader)

def test(config):

    # -------------------
    # Step 2: Define data
    # -------------------
    test_dataset = BaseAudioDataset(mode="test", num_samples=config["data"]["testing"]["num_samples"], num_time_frames=config["data"]["num_time_frames"])
    test_loader = data.DataLoader(test_dataset, batch_size=config["testing"]["batch_size"], num_workers=config["data"]["num_workers"])

    # -------------------
    # Step 3: Define model
    # -------------------
    # transformer
    latent_transformer = AV_Transformer(config["model"]["params"]["in_channels"], config["model"]["params"]["num_heads"], config["model"]["params"]["d_head"], depth=config["model"]["params"]["depth"], 
    num_layers=config["model"]["params"]["num_context_encoder_layers"],
    context_dim=config["model"]["params"]["context_dim"])
    rebalancingnet = HDemucsRemixer(sources=["output"], latent_transformer=latent_transformer)

    visah_model = VisAH(rebalancingnet, config=config, output_dir=config["testing"]["output_dir"])
    
    # Initialize trainer
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_progress_bar=True,
    )

    # Test the model
    trainer.test(visah_model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a PyTorch Lightning model using YAML config.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Replace ${model_name} with the actual model name
    model_name = config["model_name"]
    print(f"Running model {model_name}")
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = value.replace("${model_name}", model_name)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str):
                    config[key][sub_key] = sub_value.replace("${model_name}", model_name)

    # Set the seed
    seed_everything(42)  # Replace 42 with your desired seed

    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "test":
        test(config)
    else:
        raise ValueError("mode should be train or test")