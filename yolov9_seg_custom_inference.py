import os
import sys
import numpy as np
import time

import torch
import cv2


main_path =os.path.abspath(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(main_path, "..", "yolov9")))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
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
    
class YOLOV9SegInference:
    def __init__(self, config_path="inference_config/yolov9_seg_config.yaml"):
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
        self.imgsz = [640,640]

        self.load_image = LoadImages(img_size=self.imgsz, stride=stride, auto=self.pt)

        ##TODO Değiştir
        self.model_names = {"0":"0", "1": "1"}    ## class isimlerini gir
        ####

        self.cnt=0

    def __call__(self, org_image):
        start_timer = time.time()

        dataset = self.load_image(org_image)
        im, org_image = dataset

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred, self.proto = self.model(im, augment=self._config["augment"], visualize=False)[:2]  #self._config["visualize"]

        # NMS
        pred = non_max_suppression(pred, self._config["conf_thres"], self._config["iou_thres"], None,
                                        self._config["agnostic_nms"], max_det=self._config["max_det"], nm=32)
        
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        result_dict = self._detect(pred, im, org_image)

        #print("Elapsed time for segmentation: ", time.time()-start_timer)

        #self._make_binary_img(result_dict, org_image.shape)

        return result_dict # result_dict, org_image_copied
    
    @smart_inference_mode()
    def _detect(self, pred, im, org_image):
        # Process predictions
        #result_dict= []

        for i, det in enumerate(pred):  # per image
            org_image_copied = org_image.copy()

            #annotator = Annotator(org_image_copied, line_width=self._config["line_thickness"], example=str(self.names))

            masks = process_mask(self.proto[-1][i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) # HWC
            scaled_masks = self._scale_poly_points(im, masks, org_image)
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], org_image.shape).round()  # rescale boxes to im0 size

            ## mask to poly
            poly_points = self._get_seg_points(scaled_masks)
            ##

            # Write results
            self.caption = []
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                label = self.model_names[str(int(cls))]    #int(cls) #
                conf = float(conf)
                x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                #x1, y1, x2, y2 = int(x1*ratio_w), int(y1*ratio_h), int(x2*ratio_w), int(y2*ratio_h)

                matched_poly = self._match_poly_bbox(bbox1=[x1,y1,x2,y2], polygons=poly_points)

                info = dict(
                    label=label,
                    bbox=[x1,y1,x2,y2],
                    poly=matched_poly,
                    conf=conf
                )
                self.caption.append(info)

        if self._config["draw_img"]:
            org_image_copied = self._draw(org_image_copied, self.caption)

            if self._config["view_img"]:
                cv2.imshow("pred", org_image_copied)
                cv2.waitKey(0)
            
        return self.caption

    def _scale_poly_points(self, im, masks, image):
        ##calib
        org_h, org_w, _ = image.shape
        calib_w, calib_h = im.shape[2:][1], im.shape[2:][0]
        ratio_w = org_w/calib_w
        ratio_h = org_h/calib_h
        ##

        masks_np_array = masks.cpu().detach().numpy()
        new_masks = []
        for mask in masks_np_array:
            mask = mask.astype("uint8")
            mask = cv2.resize(mask, (int(mask.shape[1]*ratio_w), int(mask.shape[0]*ratio_h)))
            new_masks.append(mask)
        return np.array(new_masks)
        
    def _match_poly_bbox(self, bbox1, polygons):
        #iou_dict = {}
        result_list = []
        max_iou=0
        max_idx=-1
        for i, poly in enumerate(polygons):
            x_list = [x for i,x in enumerate(poly) if i%2==0]
            y_list = [y for i,y in enumerate(poly) if i%2!=0]
            bbox2 = min(x_list), min(y_list), max(x_list), max(y_list)

            iou = self._get_iou_from_bbox(bbox1, bbox2)
            if iou>0.0:
                #iou_dict[str(i)] = iou
                if iou>max_iou:
                    max_iou = iou
                    max_idx = i

        result_list = [int(i) for i in polygons[max_idx]]
        return result_list
    
    def _get_iou_from_bbox(self, bb1, bb2):
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        #bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[2])

        iou = (intersection_area / float(bb1_area))*100
        return iou
    
    def _fix_seg_poly(self, polygons):
        area_list = []
        for poly in polygons:
            x_list = [x for i,x in enumerate(poly) if i%2==0]
            y_list = [y for i,y in enumerate(poly) if i%2!=0]

            x1,y1,x2,y2 = min(x_list), min(y_list), max(x_list), max(y_list)
            area = (x2-x1)*(y2-y1)
            area_list.append(area)

        max_area_idx = area_list.index(max(area_list))
        return polygons[max_area_idx]
    
    def _get_seg_points(self, masks):
        import numpy as np
        from imantics import Mask

        poly_list = []
        for mask in masks:
            polygons = Mask(mask).polygons()

            poly = self._fix_seg_poly(polygons)
            poly_list.append(poly)

        return poly_list
    
    def _draw(self, image, caption):
        for info in caption:
            pts = np.array(info["poly"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            isClosed = True
            color = self.colors[int(info["label"])]
            thickness = 2
            image = cv2.polylines(image, [pts], 
                                isClosed, color, thickness)
            
        return image

    
if __name__ == "__main__":
    ai_engine = YOLOV9SegInference()

    path = ""
    dir = os.listdir(path)

    for image_name in dir:
        try:
            image = cv2.imread(os.path.join(path,image_name))
            start_timer = time.time()
            caption = ai_engine(image)
            print("Elapsed time: ", time.time()-start_timer)
        except:
            pass

        # for info in caption:
        #     pts = np.array(info["poly"], np.int32)
        #     pts = pts.reshape((-1, 1, 2))
        #     isClosed = True
        #     color = (255, 0, 0)
        #     thickness = 2
        #     image = cv2.polylines(image, [pts], 
        #                         isClosed, color, thickness)
            
        # cv2.imshow("pred", image)
        # cv2.waitKey(0)
