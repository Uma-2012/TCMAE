import glob
import os
import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

class AbnormalDatasetGradientsTest(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
            gt_path = args.avenue_gt_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
            gt_path = args.shanghai_gt_path
        else:
            raise Exception("Unknown dataset!")
        self.ds_name = args.dataset
        self.input_3d = args.input_3d
        self.data, self.labels, self.gradients = self._read_data(data_path, gt_path)

    def _read_data(self, data_path, gt_path):
        print("Initializing Test Dataset...")
        data = []
        labels = []
        gradients = []
        
        frames_base_dir = os.path.join(data_path, "testing", "frames")
        dirs = sorted(list(glob.glob(os.path.join(frames_base_dir, "*"))))

        if not dirs:
            raise FileNotFoundError(f"FATAL: No video directories found in {frames_base_dir}.")
        
        total_frames_scanned = 0
        total_frames_added = 0
        for dir_path in dirs:
            video_name = os.path.basename(dir_path)
            imgs_path = sorted(list(glob.glob(os.path.join(dir_path, "*"))))
            total_frames_scanned += len(imgs_path)
            
            label_file_path = os.path.join(gt_path, f"{video_name}.npy")
            if not os.path.exists(label_file_path):
                print(f"Warning: Missing label file {label_file_path}, skipping video {video_name}.")
                continue

            video_labels = np.load(label_file_path)

            for i, img_path in enumerate(imgs_path):
                frame_basename = os.path.splitext(os.path.basename(img_path))[0]
                frame_number = int(frame_basename)
                gradient_p = os.path.join(data_path, "testing", "gradients2", video_name, f"{frame_number:03d}.jpg")

                if os.path.exists(gradient_p) and i < len(video_labels):
                    data.append(img_path)
                    gradients.append(gradient_p)
                    labels.append(video_labels[i])
                    total_frames_added += 1
        
        print(f"Dataset Initialized: Scanned {total_frames_scanned} frames, added {total_frames_added} complete samples.")
        return data, labels, gradients

    def __getitem__(self, index):
        target_w, target_h = 160, 320
        target_size_wh = (target_w, target_h)

        def read_and_resize(path):
            img = cv2.imread(path)
            if img is None:
                raise IOError(f"FATAL: Could not read image at path: {path}.")
            return cv2.resize(img, target_size_wh)

        current_img = read_and_resize(self.data[index])
        # Load but don't use in input
        gradient = read_and_resize(self.gradients[index])

        img = current_img
        target = current_img

        if self.input_3d:
            dir_path, frame_no, len_frame_no, ext = self.extract_meta_info(self.data[index])
            k = 3
            previous_img = self.read_prev_next_frame(dir_path, frame_no, ext, direction=-k, length=len_frame_no, target_size_wh=target_size_wh)
            next_img = self.read_prev_next_frame(dir_path, frame_no, ext, direction=k, length=len_frame_no, target_size_wh=target_size_wh)
            
            # Input Construction: [Prev, Curr, Next] -> 9 Channels
            img = np.concatenate([previous_img, current_img, next_img], axis=-1)
        
        # Preprocessing
        img = (img.astype(np.float32) - 127.5) / 127.5
        target = (target.astype(np.float32) - 127.5) / 127.5
        
        dummy_grad_mask = np.zeros_like(target)

        img = np.transpose(img, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))
        dummy_grad_mask = np.transpose(dummy_grad_mask, (2, 0, 1))
        
        return img, dummy_grad_mask, target, self.labels[index], self.data[index].split('/')[-2], self.data[index]

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