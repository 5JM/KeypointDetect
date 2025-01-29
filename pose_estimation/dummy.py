# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, glob
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import cv2
import numpy as np

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models
from models.pose_resnet import get_pose_net
import json
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    data_list = ['train', 'valid', 'test']
    for name in data_list:
        with open(os.path.join('data',name,'_annotations.coco.json'), 'r') as f:
            tmp = json.load(f)

        for i in tmp['annotations']:
            if i['category_id'] == 1:
                i['category_id'] = 0

        with open(os.path.join('data',name,'_annotations.coco.json'), 'w') as f:
            json.dump(tmp, f, indent="\t")

class TestDataset(Dataset):
    def __init__(self, annotation_file, image_dir, target_size=640, transform=None, gray = True):
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.gray = gray
        self.image_dir = image_dir
        self.target_size = target_size
        self.transform = transform
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # Create a mapping from image id to annotations
        self.img_id_to_annotations = {}
        for annotation in self.annotations:
            img_id = annotation['image_id']
            if img_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[img_id] = []
            self.img_id_to_annotations[img_id].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.image_dir, img_info['file_name'].split('/')[-1])
        # Load image in grayscale
        # print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        H,W = image.shape[:2]

        im = np.zeros((H,W,3), dtype= np.uint8)
        im[:,:,0] = image
        im[:,:,1] = image
        im[:,:,2] = image

        keypoints, bboxes = [], []
        if img_id in self.img_id_to_annotations:
            for annotation in self.img_id_to_annotations[img_id]:
                keypoints.append(annotation['keypoints'])
                bboxes.append(annotation['bbox'])  # [x, y, width, height]

        # Convert keypoints to numpy array
        keypoints = np.array(keypoints)
        bboxes = np.array(bboxes)
        # print(keypoints.shape)
        return image if self.gray else im, keypoints[0], bboxes[0], img_path  # Return original image and empty keypoints if no bbox
    
        # Crop the image based on the first bbox (you can modify this as needed)
        if len(bboxes) > 0:
            x, y, w, h = map(int, bboxes[0])
            
            cropped_image = im[y:y+h, x:x+w, :]
            print(img_path)
            print(cropped_image.shape, x, y, w, h)
            # Resize while maintaining the aspect ratio
            h_cropped, w_cropped = cropped_image.shape[:2]
            scale = min(self.target_size / w_cropped, self.target_size / h_cropped)
            new_w, new_h = int(w_cropped * scale), int(h_cropped * scale)

            resized_image = cv2.resize(cropped_image, (new_w, new_h))
            padded_image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            pad_x = (self.target_size - new_w) // 2
            pad_y = (self.target_size - new_h) // 2
            padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image

            # Adjust keypoints based on the padded image
            adjusted_keypoints =[]

            # keypoints = np.squeeze(keypoints)
            
            keypoints = keypoints.reshape((-1, 3))
            for kp in keypoints:
                adjusted_kp = kp[:2] - [x, y]  # 크롭된 이미지의 기준으로 변환
                adjusted_kp *= scale  # 스케일 적용
                adjusted_kp[0] += pad_x  # 패딩 추가
                adjusted_kp[1] += pad_y  # 패딩 추가
                
                adjusted_keypoints.append(np.hstack((adjusted_kp, kp[2:])))  # 가시성 정보 추가

            adjusted_keypoints = np.array(adjusted_keypoints)
            
            if self.transform:
                padded_image = self.transform(padded_image)
                
            return padded_image, adjusted_keypoints, img_path
        else:
            return image, keypoints, img_path  # Return original image and empty keypoints if no bbox
        

    def visualize(self, image, keypoints, bboxes, image_path):
        for i, annotation in enumerate(keypoints):
            # Draw keypoints
            x, y, v = annotation
            if v > 0:  # Only draw visible keypoints
                cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255), -1)
            else:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

            box_x, box_y, box_w, box_h = bboxes
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_x + box_w), int(box_y + box_h)),(255,255,255), 2)

            # Draw keypoint ID next to the keypoint
            keypoint_id = i  # Assuming keypoint IDs are sequential
            cv2.putText(image, str(keypoint_id), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow(f'{image_path[0].split('/')[-1]}', image)

def check_data(img_list_path, label_path):
    # img_list_path = '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/crop_pad'
    # label_path = '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/seat_keypoints_default.json'

    assert os.path.exists(img_list_path), 'No exist image foler'
    assert os.path.exists(label_path), 'No exist label file'

    dataset = TestDataset(label_path, img_list_path, gray = True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Visualize keypoints for each batch
    for images, keypoints, bboxes, img_path in dataloader:
        for i in range(images.size(0)):  # Loop through the batch
            img = images[i].numpy()  # Convert to numpy for visualization
            # img = img.squeeze()  # Remove channel dimension if necessary
            # img = img.transpose(1, 2, 0)
            img = img.astype(np.uint8)  # Convert to uint8 for OpenCV
            # print(img.shape)
            # Visualize keypoints
            dataset.visualize(img, keypoints[i].numpy(), bboxes[i].numpy(), img_path)
            
            # Wait for a key press and check for ESC
            key = cv2.waitKey(0)
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                break

            cv2.destroyAllWindows()
        else:
            continue  # Only executed if the inner loop did NOT break
        break  # Break the outer loop if ESC was pressed

def split_coco_dataset(coco_file, image_dir, output_dir='./split_coco', train_ratio=0.8, val_ratio=0.1):
    # COCO JSON 파일 읽기
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # 이미지와 어노테이션 리스트 추출
    images = coco_data['images']
    annotations = coco_data['annotations']

    # 이미지 ID를 기준으로 train, val, test 데이터셋 분할
    image_ids = [img['id'] for img in images]
    train_ids, temp_ids = train_test_split(image_ids, train_size=train_ratio)
    val_ids, test_ids = train_test_split(temp_ids, test_size=val_ratio/(1 - train_ratio))

    # 출력 디렉토리 생성
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # 새로운 COCO 데이터셋 구조 초기화
    train_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']}
    val_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']}
    test_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']}

    # 각 데이터셋에 이미지와 어노테이션 추가
    for img in images:
        img_id = img['id']
        img_file = os.path.join(image_dir, img['file_name'])
        # print(img_file)
        # print(os.path.join(output_dir, 'train', img['file_name']))
        if img_id in train_ids:
            train_data['images'].append(img)
            shutil.copy(img_file, os.path.join(output_dir, 'train', img['file_name']))
        elif img_id in val_ids:
            val_data['images'].append(img)
            shutil.copy(img_file, os.path.join(output_dir, 'val', img['file_name']))
        elif img_id in test_ids:
            test_data['images'].append(img)
            shutil.copy(img_file, os.path.join(output_dir, 'test', img['file_name']))

    for ann in annotations:
        if ann['image_id'] in train_ids:
            train_data['annotations'].append(ann)
        elif ann['image_id'] in val_ids:
            val_data['annotations'].append(ann)
        elif ann['image_id'] in test_ids:
            test_data['annotations'].append(ann)

    # 어노테이션 파일 저장
    with open(os.path.join(output_dir, 'train', 'annotations.json'), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, 'val', 'annotations.json'), 'w') as f:
        json.dump(val_data, f)
    with open(os.path.join(output_dir, 'test', 'annotations.json'), 'w') as f:
        json.dump(test_data, f)

    print("데이터셋 분할 완료: train, val, test 디렉토리 및 어노테이션 파일이 생성되었습니다.")

def crop_and_pad_coco_images(coco_file, image_dir, output_dir, target_size=(640, 640)):
    # COCO JSON 파일 읽기
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    os.makedirs(output_dir, exist_ok=True)

    new_images = []
    new_annotations = []

    for img in images:
        img_id = img['id']
        img_file = os.path.join(image_dir, img['file_name'])

        # 이미지 읽기
        image = cv2.imread(img_file)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {img_file}")
            continue

        # 해당 이미지의 bbox와 keypoints 찾기
        img_annotations = [ann for ann in annotations if ann['image_id'] == img_id]

        for ann in img_annotations:
            bbox = ann['bbox']
            keypoints = ann['keypoints']

            # bbox 크롭 좌표 계산
            x, y, w, h = map(int, bbox)
            margin = 50
            x -= margin
            y -= margin
            margin = 150
            w += margin
            h += margin

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            crop_img = image[y:y+h, x:x+w]

            # 크롭된 이미지의 비율 유지 및 패딩
            crop_height, crop_width = crop_img.shape[:2]
            scale = min(target_size[1] / crop_width, target_size[0] / crop_height)
            new_w, new_h = int(crop_width * scale), int(crop_height * scale)

            resized_image = cv2.resize(crop_img, (new_w, new_h))
            padded_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

            pad_x = (target_size[1] - new_w) // 2
            pad_y = (target_size[0] - new_h) // 2

            padded_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image

            adjust_bboxes = [pad_x, pad_y, new_w, new_h]
            # 키포인트 조정
            adjusted_keypoints = []

            keypoints = np.array(keypoints)
            keypoints = keypoints.reshape((-1, 3))

            for kp in keypoints:
                adjusted_kp = kp[:2] - [x, y]  # 크롭된 이미지의 기준으로 변환
                adjusted_kp *= scale  # 스케일 적용
                adjusted_kp[0] += pad_x  # 패딩 추가
                adjusted_kp[1] += pad_y  # 패딩 추가
                
                _tmp = np.hstack((adjusted_kp, kp[2:]))

                adjusted_keypoints.append(_tmp.tolist())  # 가시성 정보 추가
            # print(type(adjusted_keypoints))
            # adjusted_keypoints = np.array(adjusted_keypoints)
            
            # flatten
            adjusted_keypoints = [item for sub_list in adjusted_keypoints for item in sub_list]

            # 새로운 이미지 정보 추가
            new_image_info = {
                'id': len(new_images) + 1,
                'file_name': img['file_name'].split('/')[-1],
                'width': target_size[1],
                'height': target_size[0]
            }
            new_images.append(new_image_info)

            # 어노테이션 업데이트
            new_annotation = {
                'id': len(new_annotations) + 1,  # 고유한 ID 할당
                'image_id': new_image_info['id'],
                'bbox': adjust_bboxes,
                'keypoints': adjusted_keypoints,
                'area': w * h,  # 필요에 따라 조정
                'iscrowd': 0,
                'category_id': ann['category_id']
            }
            new_annotations.append(new_annotation)

            # 크롭된 이미지 저장
            output_image_file = os.path.join(output_dir, img['file_name'].split('/')[-1])
            cv2.imwrite(output_image_file, padded_img)

    # 새로운 COCO 형식의 JSON 생성
    new_coco_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': categories
    }

    # 새로운 어노테이션 파일 저장
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as ann_file:
        json.dump(new_coco_data, ann_file)

    print("이미지 크롭 및 패딩 완료.")

if __name__ == '__main__':
    # main()

    # crop_and_pad_coco_images(
    #     coco_file = '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/seat_keypoints_default.json', 
    #     image_dir = '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/', 
    #     output_dir = '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/crop_pad_320/', 
    #     target_size=(320, 320)
    #     )

    # check_data(
    #     img_list_path = '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/crop_pad_320/', 
    #     label_path= '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/crop_pad_320/annotations.json', 
    # )

    split_coco_dataset(
        coco_file='/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/crop_pad_320/annotations.json',
        image_dir='/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/crop_pad_320/',
        output_dir = '/home/jaemu/Desktop/human-pose-estimation.pytorch/pose_estimation/'
    )
