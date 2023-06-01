from tqdm import tqdm
import numpy as np
import h5py
import shutil
from scenedetect import detect, AdaptiveDetector
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
from PIL import Image
import os
import cv2
import csv

FRAME_RATE = 15

class Rescale(object):
    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


net = models.googlenet(pretrained=True).float()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])

class GenerateDataset:
    def __init__(self, video_path, save_path, video_count):
        self.video_count = video_count
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.frame_root_path = 'frames'
        self.h5_file = h5py.File(save_path, 'w')

        self._set_video_list(video_path)

    def _set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = []
            for filename in os.listdir(video_path):
                if filename.endswith(".mp4"):
                    self.video_list.append(filename)
            self.video_list = sorted(self.video_list, key=lambda x: int(x.split(".")[0]))
            if self.video_count < len(self.video_list):
                self.video_list = self.video_list[:self.video_count]
            print(f"{len(self.video_list)} videos: {self.video_list}")
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))


    def _extract_feature(self, frame):
        res_pool5 = fea_net(transform(Image.fromarray(frame)).unsqueeze(0)).squeeze().detach().cpu()
        frame_feat = res_pool5.cpu().data.numpy().flatten()

        return frame_feat

    def _get_change_points(self, video):
        scene_list = detect(video, AdaptiveDetector())
        change_points = []
        n_frame_per_seg = []
        for i, scene in enumerate(scene_list):
            f0 = scene[0].get_frames()
            f1 = scene[1].get_frames()-1
            change_points.append((f0, f1))
            n_frame_per_seg.append(f1 - f0 + 1)
        return change_points, n_frame_per_seg

    def generate_dataset(self):
        for video_idx, video_filename in enumerate(tqdm(self.video_list)):
            print(f"\nvideo {video_idx+1}, {video_filename}")
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_basename = os.path.basename(video_path).split('.')[0]
            
            frame_directory = os.path.join(self.frame_root_path, video_basename)
            if not os.path.exists(frame_directory):
                os.mkdir(frame_directory)

            video_capture = cv2.VideoCapture(video_path)

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            picks = []
            video_feat_for_train = None
            for frame_idx in tqdm(range(n_frames)):
                success, frame = video_capture.read()
                if frame_idx % FRAME_RATE == 0:
                    if success:
                        frame_feat = self._extract_feature(frame)

                        picks.append(frame_idx)

                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))

                        img_filename = "{}.jpg".format(str(frame_idx).zfill(5))
                        cv2.imwrite(os.path.join(self.frame_root_path, video_basename, img_filename), frame)

                    else:
                        break

            video_capture.release()

            change_points, n_frame_per_seg = self._get_change_points(video_path)
            shutil.rmtree(frame_directory)
            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg

            n = FRAME_RATE #fps
            l = 15
            fileName = 'data/'+ str(video_idx+1) + "_heatmap_final.csv"

            file = open(fileName,"r")
            data = np.array(list(csv.reader(file, delimiter=",")))
            file.close()
            data = data[:,1]
            data = data[1:]
            data = np.array(data, dtype=float)
            data -= data.min()
            data /= data.max()
            
            n_frames = int(data.shape[0])
            #n_steps = int(n_frames//n + 1)
            #n_steps = int(n_frames/n)
            n_steps = len(picks)

            # Compute gt score of each segment
            gtscore = np.zeros(n_steps, dtype=float)
            for k in range(n_steps):
                gtscore[k] = np.sum(data[k*n:((k+1)*n)]) / n


            # Compute gt summary binary value using basic knapsack
            num_p = n_steps * l // 100
            idx = np.argpartition(gtscore, -num_p)[-num_p:]
            gtsummary = np.zeros(n_steps, dtype=float)
            gtsummary[idx] = 1.0

            user_summary = np.ones((20, n_frames)) * data

            self.h5_file['video_{}'.format(video_idx+1)]['gtsummary'] = gtsummary
            self.h5_file['video_{}'.format(video_idx+1)]['gtscore'] = gtscore
            self.h5_file['video_{}'.format(video_idx+1)]['user_summary'] = user_summary