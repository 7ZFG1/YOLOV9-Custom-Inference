# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch

import sys
import cv2

main_path =os.path.abspath(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(main_path, "..", "yolov9")))   ##yolov9 folder'ı
#sys.path.insert(0,os.path.abspath(os.path.join(main_path, "..", "yolov9")))

from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from utils.augmentations import letterbox

from inference_config.args import parse_args

class LoadImages:
    def __init__(self, img_size=640, stride=32, auto=True):
        self.img_size = img_size
        self.stride = stride
        self.auto = auto

    def __call__(self, org_image):
        ###im0 ---> org_image
        im = letterbox(org_image, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        return im, org_image
    
class YOLOV9Inference:
    def __init__(self, config_path="inference_config/yolov9_config.yaml"):
        self._config = parse_args(config_path)

        self.colors = [(255,0,0), (0,255,0), (0,0,255),(255,255,0),
                    (0,255,255),(255,0,255),(125,0,255),
                    (0,0,0),(255,255,255)] #[(255,0,0), (0,255,0), (0,0,255),(255,255,0),(0,255,255),(255,0,255),(125,0,255)]

        device = select_device(self._config["device"])
        self.model = DetectMultiBackend(self._config["weights"], device=device, dnn=self._config["dnn"], 
                                   data=self._config["data"], fp16=self._config["half"])
        stride, self.names, self.pt = self.model.stride, self.model.names, self.model.self.pt
        self.imgsz = (self._config["imgsz"], self._config["imgsz"])
        self.imgsz = check_img_size(self._config["imgsz"], s=stride)  # check image size

        self.load_image = LoadImages(img_size=self.imgsz, stride=stride, auto=self.pt)
       
    def __call__(self, org_image):
        dataset = self.load_image(org_image)
        im, org_image = dataset

        bs=1
        #self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        self.seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=self._config["augment"], visualize=False)  #self._config["visualize"]

        # NMS
        pred = non_max_suppression(pred, self._config["conf_thres"], self._config["iou_thres"], self._config["classes"], 
                                        self._config["agnostic_nms"], self._config["multi_label"], max_det=self._config["max_det"])
        
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        result_dict, org_image_copied = self._detect(pred, im, org_image)

        return result_dict # result_dict, org_image_copied

        # t = tuple(x.t / self.seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)

    @smart_inference_mode()
    def _detect(self, pred, im, org_image):
        # Process predictions
        result_dict= []

        for i, det in enumerate(pred):  # per image
            self.seen += 1

            org_image_copied = org_image.copy()

            # gn = torch.tensor(org_image_copied.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = org_image_copied.copy() if self._config["save_crop" ]else org_image_copied  # for save_crop
            annotator = Annotator(org_image_copied, line_width=self._config["line_thickness"], example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to org_image_copied size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], org_image_copied.shape).round()

                # # # Print results
                # # for c in det[:, 5].unique():
                # #     n = (det[:, 5] == c).sum()  # detections per class
                # #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = self.names[int(cls)]
            
                    conf = float(conf)
                    x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    info = dict(
                        label=label,
                        bbox=[x1,y1,x2,y2],
                        conf=conf
                    )

                    result_dict.append(info)


                    if self._config["draw_img"]:  # Add bbox and text to image
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])

        org_image_copied = annotator.result()
        if self._config["save_img"]:
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path,fname), org_image_copied)
            print("Saved: ", save_path+fname)

        if self._config["view_img"]:
            cv2.imshow("Result", cv2.resize(org_image_copied, (int(org_image_copied.shape[1]/4), int(org_image_copied.shape[0]/4))))
            cv2.waitKey(0)

        return result_dict, org_image_copied
    
if __name__ == "__main__":
    import time
    engine = YOLOV9Inference()

    image_path = ""
    image_listdir = os.listdir(image_path)
    save_path = ""

    for fname in sorted(image_listdir):
        full_image_path = os.path.join(image_path, fname)
        image = cv2.imread(full_image_path)

        if not type(image) == np.ndarray:
            continue

        s = time.time()
        result_dict = engine(image)
        print("Elapsed Time: ", time.time()-s)

        label_dict = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0}
        for info in result_dict:
            label = info["label"]
            label_dict[label] += 1

        print("IMAGE NAME: ", fname)
        print(label_dict)
        print("-------------")
