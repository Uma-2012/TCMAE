import glob
import os
import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

class AbnormalDatasetGradientsTrain(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
        else:
            raise Exception("Unknown dataset!")
        
        self.input_3d = args.input_3d
        self.data, self.gradients = self._read_data(data_path)

    def _read_data(self, data_path):
        print("Initializing Unsupervised Train Dataset...")
        data = []
        gradients = []
        
        extension = None
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "training/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break
        
        if not extension:
            raise FileNotFoundError("No image files found in the training/frames directory.")

        dirs = sorted(list(glob.glob(os.path.join(data_path, "training", "frames", "*"))))
        
        for dir_path in dirs:
            video_name = os.path.basename(dir_path)
            imgs_path = sorted(list(glob.glob(os.path.join(dir_path, f"*{extension}"))))
            
            for img_path in imgs_path:
                frame_filename = os.path.basename(img_path)
                gradient_p = os.path.join(data_path, "training", "gradients2", video_name, frame_filename)
                
                if os.path.exists(gradient_p):
                    data.append(img_path)
                    gradients.append(gradient_p)
        
        print(f"Dataset Initialized: Found {len(data)} normal samples.")
        return data, gradients

    def __getitem__(self, index):
        target_w, target_h = 160, 320
        target_size_wh = (target_w, target_h)

        def read_and_resize(path):
            img = cv2.imread(path)
            if img is None:
                raise IOError(f"FATAL: Could not read image at path: {path}.")
            return cv2.resize(img, target_size_wh)

        img_path = self.data[index]
        current_img = read_and_resize(img_path)
        
        # Load gradient just in case, though unused in input/loss
        gradient_path = self.gradients[index]
        gradient = read_and_resize(gradient_path)
        
        # Target is the current RGB frame
        target = current_img

        # Input Construction: [I_{t-k}, I_t, I_{t+k}] -> 9 Channels
        img = current_img
        if self.input_3d:
            dir_path, frame_no, len_frame_no, ext = self.extract_meta_info(img_path)
            # Using k=3 (step 3) as per original code logic, can be adjusted to k=1 if needed
            k = 3
            previous_img = self.read_prev_next_frame(dir_path, frame_no, ext, direction=-k, length=len_frame_no, target_size_wh=target_size_wh)
            next_img = self.read_prev_next_frame(dir_path, frame_no, ext, direction=k, length=len_frame_no, target_size_wh=target_size_wh)
            
            img = np.concatenate([previous_img, current_img, next_img], axis=-1)

        # Preprocessing
        img = (img.astype(np.float32) - 127.5) / 127.5
        target = (target.astype(np.float32) - 127.5) / 127.5
        
        # Pass dummy mask since we removed gradient weighting
        dummy_grad_mask = np.zeros_like(target)

        img = np.transpose(img, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))
        dummy_grad_mask = np.transpose(dummy_grad_mask, (2, 0, 1))
        
        return img, dummy_grad_mask, target

    def extract_meta_info(self, img_path):
        dir_path = os.path.dirname(img_path)
        base, ext = os.path.splitext(os.path.basename(img_path))
        frame_no_str = base
        frame_no_int = int(frame_no_str)
        return dir_path, frame_no_int, len(frame_no_str), ext

    def read_prev_next_frame(self, dir_path, frame_no, ext, direction, length, target_size_wh):
        nearby_frame_no_str = str(frame_no + direction).zfill(length)
        frame_path = os.path.join(dir_path, f"{nearby_frame_no_str}{ext}")

        if not os.path.exists(frame_path):
            current_frame_no_str = str(frame_no).zfill(length)
            frame_path = os.path.join(dir_path, f"{current_frame_no_str}{ext}")

        img = cv2.imread(frame_path)
        if img is None:
            raise IOError(f"FATAL: Could not read nearby frame at path: {frame_path}")
        
        return cv2.resize(img, target_size_wh)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__