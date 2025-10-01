"""
Model inference script for acoustic highlighting.
This script processes videos to generate enhanced audio using visual features.
"""

import torch
import torch.nn as nn
import torchaudio
import os
import argparse
import yaml
import logging
from typing import Tuple, Optional
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from models.audio_net import (
    SpectrogramUnetUnified, SpectrogramUnetNoCond, AudioVisualMaskFormer
)
from models.hdemucs import HDemucsRemixer
from models.model import SpectrogramNet
from models.attention import ZeroTransformer, AV_Transformer
from pytorch_lightning import seed_everything

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoAudioProcessor:
    """Handles video and audio processing operations."""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", cache_dir: str = None):
        """
        Initialize the processor with CLIP model.
        
        Args:
            model_name: HuggingFace model identifier for CLIP
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.clip_model = None
        self.clip_processor = None
    
    def load_clip_model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load CLIP model and processor from HuggingFace transformers."""
        try:
            processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            model = CLIPModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.clip_processor = processor
            self.clip_model = model
            logger.info(f"Successfully loaded CLIP model: {self.model_name}")
            return model, processor
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def extract_frames_and_audio(self, video_path: str, target_sr: int = 44100, 
                                batch_size: int = 32) -> Tuple[torch.Tensor, np.ndarray, int]:
        """
        Extract frames (1 FPS) and audio from an MP4 video file.
        
        Args:
            video_path: Path to the input MP4 file
            target_sr: Target sampling rate for audio
            batch_size: Batch size for processing frames
        
        Returns:
            Tuple of (processed_frames, audio_data, sampling_rate)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if self.clip_processor is None:
            self.load_clip_model()
        
        # Extract frames using OpenCV
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps))  # Ensure at least 1
        current_frame = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Extract one frame per second
                if current_frame % frame_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    frames.append(pil_image)
                    
                current_frame += 1
        finally:
            cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Process frames with CLIP processor in batches
        processed_frames = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
            processed_frames.append(inputs['pixel_values'])
        
        processed_frames = torch.cat(processed_frames, dim=0)
        
        # Extract audio using moviepy
        try:
            video_clip = VideoFileClip(str(video_path))
            audio_clip = video_clip.audio
            
            if audio_clip is None:
                raise ValueError("No audio track found in video")
            
            audio_data = audio_clip.to_soundarray()
            original_sr = int(audio_clip.fps)
            
            # Convert stereo to mono if necessary
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            
        finally:
            if 'video_clip' in locals():
                video_clip.close()
            if 'audio_clip' in locals():
                audio_clip.close()
        
        return processed_frames, audio_data, original_sr
    
    def process_video_for_clip(self, video_path: str, device: str = "cuda") -> Tuple[torch.Tensor, np.ndarray, int]:
        """
        Process video and extract CLIP features.
        
        Args:
            video_path: Path to input video
            device: Device to run inference on
            
        Returns:
            Tuple of (image_features, normalized_audio, sampling_rate)
        """
        if self.clip_model is None:
            self.load_clip_model()
        
        self.clip_model = self.clip_model.to(device)
        
        # Extract and process frames and audio
        frames_tensor, audio_data, sr = self.extract_frames_and_audio(video_path)
        
        # Get CLIP image features
        with torch.no_grad():
            frames_tensor = frames_tensor.to(device)
            image_features = self.clip_model.get_image_features(frames_tensor)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu(), audio_data, sr
    
    def replace_video_audio(self, video_path: str, new_audio_path: str, 
                          output_path: str, duration: float = 9.0) -> None:
        """
        Replace the audio of a video with a new audio track.
        
        Args:
            video_path: Path to the original video
            new_audio_path: Path to the new audio file
            output_path: Path where to save the new video
            duration: Duration to clip both video and audio to
        """
        try:
            # Load the original video and new audio
            video = VideoFileClip(video_path)
            new_audio = AudioFileClip(new_audio_path)
            
            # Clip to specified duration
            new_audio = new_audio.subclip(0, min(duration, new_audio.duration))
            video = video.subclip(0, min(duration, video.duration))
            
            # Create final video with replaced audio
            final_video = video.set_audio(new_audio).set_duration(duration)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the result to file
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None  # Suppress moviepy logs
            )
            
            logger.info(f"Video saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error replacing video audio: {e}")
            raise
        finally:
            # Clean up
            if 'video' in locals():
                video.close()
            if 'new_audio' in locals():
                new_audio.close()
            if 'final_video' in locals():
                final_video.close()


def load_model(config: dict, device: torch.device) -> HDemucsRemixer:
    """
    Load and initialize the model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model ready for inference
    """
    try:
        # Initialize transformer and model
        latent_transformer = AV_Transformer(
            config["model"]["params"]["in_channels"], 
            config["model"]["params"]["num_heads"], 
            config["model"]["params"]["d_head"], 
            depth=config["model"]["params"]["depth"], 
            context_dim=config["model"]["params"]["context_dim"]
        )
        
        model = HDemucsRemixer(sources=["output"], latent_transformer=latent_transformer).to(device)
        
        # Load checkpoint
        checkpoint_path = config["testing"]["checkpoint_path"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        
        # Clean up state dict keys
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if "loss" not in k}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_audio(audio: np.ndarray, sr: int, target_length_sec: float = 10.0, 
                    target_sr: int = 44100) -> torch.Tensor:
    """
    Preprocess audio for model input.
    
    Args:
        audio: Input audio array
        sr: Current sampling rate
        target_length_sec: Target length in seconds
        target_sr: Target sampling rate
        
    Returns:
        Preprocessed audio tensor
    """
    # Resample if necessary
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    target_samples = int(target_length_sec * target_sr)
    
    # Pad or trim to target length
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    else:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    
    return torch.from_numpy(audio).float()


def run_inference(config: dict) -> None:
    """
    Main inference function.
    
    Args:
        config: Configuration dictionary
    """
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Running model: {config['model_name']}")
    
    # Initialize processor and load model
    processor = VideoAudioProcessor(cache_dir=config.get("cache_dir"))
    model = load_model(config, device)
    
    video_path = config["testing"]["video_path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    
    try:
        # Process video
        logger.info("Processing video...")
        frames_tensor, normalized_audio, sr = processor.process_video_for_clip(video_path, device)
        
        logger.info(f"Original audio length: {len(normalized_audio) / sr:.2f} seconds, sr: {sr}")
        
        # Preprocess audio
        audio_tensor = preprocess_audio(normalized_audio, sr)
        
        # Prepare tensors for model
        frames_tensor = frames_tensor.unsqueeze(0).to(device)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(1).to(device)
        
        logger.info("Running model inference...")
        
        # Model inference
        with torch.no_grad():
            pred_audio, gt_mask, pred_mask, weight = model(audio_tensor, audio_tensor, frames_tensor)
        
        # Post-process output
        pred_audio = pred_audio.squeeze(0).cpu()
        original_audio = audio_tensor.squeeze(0).cpu()
        
        # Resample back to original sample rate if needed
        if sr != 44100:
            pred_audio = torchaudio.transforms.Resample(44100, sr)(pred_audio)
            original_audio = torchaudio.transforms.Resample(44100, sr)(original_audio)
        
        # Trim to original length if it was shorter than 10 seconds
        original_length_samples = len(normalized_audio)
        if len(normalized_audio) < sr * 10:
            pred_audio = pred_audio[:, :original_length_samples]
            original_audio = original_audio[:, :original_length_samples]
        
        # Ensure output directories exist
        for path_key in ["audio_output_path", "original_audio_output_path", "video_output_path", "original_video_output_path"]:
            output_path = config["testing"][path_key]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save audio files
        logger.info("Saving audio files...")
        torchaudio.save(config["testing"]["original_audio_output_path"], original_audio, sr)
        torchaudio.save(config["testing"]["audio_output_path"], pred_audio, sr)
        
        # Create videos with replaced audio
        logger.info("Creating output videos...")
        processor.replace_video_audio(
            video_path=video_path,
            new_audio_path=config["testing"]["audio_output_path"],
            output_path=config["testing"]["video_output_path"]
        )
        
        processor.replace_video_audio(
            video_path=video_path,
            new_audio_path=config["testing"]["original_audio_output_path"],
            output_path=config["testing"]["original_video_output_path"]
        )
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


def substitute_config_variables(config: dict, model_name: str) -> dict:
    """
    Replace ${model_name} placeholders in configuration.
    
    Args:
        config: Configuration dictionary
        model_name: Model name to substitute
        
    Returns:
        Updated configuration dictionary
    """
    def substitute_recursive(obj):
        if isinstance(obj, str):
            return obj.replace("${model_name}", model_name)
        elif isinstance(obj, dict):
            return {k: substitute_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_recursive(item) for item in obj]
        else:
            return obj
    
    return substitute_recursive(config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run acoustic highlighting model inference on video files."
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the YAML config file'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Load configuration
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        # Substitute variables
        model_name = config["model_name"]
        config = substitute_config_variables(config, model_name)
        
        # Run inference
        run_inference(config)
        
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        raise


if __name__ == "__main__":
    main()