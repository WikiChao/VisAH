from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os
import warnings
warnings.filterwarnings("ignore")

class IB_score_metric(torch.nn.Module):
    def __init__(self, device):
        super(IB_score_metric, self).__init__()
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        print('IB_score model loaded')
        self.model.eval()
        self.model.to(device)

    def load_data(self, output_dir, folder_path, device):
        folder_name = os.path.basename(folder_path)
        image_path_root = os.path.join(folder_path, 'frames')
        image_paths = [os.path.join(image_path_root, f) for f in os.listdir(image_path_root) if f.endswith('.png')]
        # sort image paths
        image_paths = sorted(image_paths)
        input_audio_path = os.path.join(output_dir, folder_name, f"input_audio.wav")
        pred_audio_path = os.path.join(output_dir, folder_name, f"pred_audio.wav")
        target_audio_path = os.path.join(output_dir, folder_name, f"target_audio.wav")
        # Load data
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([input_audio_path, pred_audio_path, target_audio_path], device),
        }
        
        return inputs

    def forward(self, output_dir, folder_path, device):
        inputs = self.load_data(output_dir, folder_path, device)
        with torch.no_grad():
            embeddings = self.model(inputs)
        # return (embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T).mean(dim=0)
        # calculate cosine similarity between vision and audio embeddings
        vision_embedding = embeddings[ModalityType.VISION]
        audio_embedding = embeddings[ModalityType.AUDIO]
        vision_embedding = vision_embedding / vision_embedding.norm(dim=-1, keepdim=True)
        audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
        return (vision_embedding @ audio_embedding.T).mean(dim=0)
    

    def forward_path(self, image_path, audio_path, device):
        image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png')]
        image_paths = sorted(image_paths)
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path,], device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        # return (embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T).mean(dim=0)
        # calculate cosine similarity between vision and audio embeddings
        vision_embedding = embeddings[ModalityType.VISION]
        audio_embedding = embeddings[ModalityType.AUDIO]
        vision_embedding = vision_embedding / vision_embedding.norm(dim=-1, keepdim=True)
        audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
        return (vision_embedding @ audio_embedding.T).mean(dim=0)
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ib_score = IB_score_metric(device)
    # output_dir = '/u/chuang65/project/acoustic highlight/experiments/acoustic highlighting/HDemucsAttnMagmask-M(nonorm)+T(norm)-TContextEnc0/test_outputs'
    # folder_path = '/home/cxu-serve/p62/chuang65/smart remixing/CondensedMovies_processed_Action/nrizm2gnQBo/sub_video/nrizm2gnQBo_012'
    # ibs = ib_score.forward(output_dir, folder_path, device)

    image_paths = "/u/chuang65/project/acoustic highlight/experiments/v2a/moviegen_ours/AQOPSRhPlINHRSMKMJjdm7LaAR3h329xG5jNuPcmfu6kfARzUp5Yeqgm0RA9Wb9oMYJPgP37ZrR91Dw8mqXTJ-2v"
    audio_path = "/u/chuang65/project/acoustic highlight/experiments/v2a/moviegen_ours/AQOPSRhPlINHRSMKMJjdm7LaAR3h329xG5jNuPcmfu6kfARzUp5Yeqgm0RA9Wb9oMYJPgP37ZrR91Dw8mqXTJ-2v.wav"
    ibs = ib_score.forward_path(image_paths, audio_path, device)

    print('IB_score done')
    print(ibs)




    