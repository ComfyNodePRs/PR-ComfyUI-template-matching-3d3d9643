# https://github.com/cozheyuanzhangde/Invariant-TemplateMatching
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import logging
import torch
from PIL import Image

logger = logging.getLogger('comfyui_template_matching')

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = 0
    if max_percent_width < max_percent_height:
        max_percent = max_percent_width
    else:
        max_percent = max_percent_height
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return result, percent

def invariant_match_template(rgbimage, rgbtemplate, method, matched_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax, rgbdiff_thresh=float("inf")):
    """
    rgbimage: RGB image where the search is running.
    rgbtemplate: RGB searched template. It must be not greater than the source image and have the same data type.
    method: [String] Parameter specifying the comparison method
    matched_thresh: [Float] Setting threshold of matched results(0~1).
    rot_range: [Integer] Array of range of rotation angle in degrees. Example: [0,360]
    rot_interval: [Integer] Interval of traversing the range of rotation angle in degrees.
    scale_range: [Integer] Array of range of scaling in percentage. Example: [50,200]
    scale_interval: [Integer] Interval of traversing the range of scaling in percentage.
    rm_redundant: [Boolean] Option for removing redundant matched results based on the width and height of the template.
    minmax:[Boolean] Option for finding points with minimum/maximum value.
    rgbdiff_thresh: [Float] Setting threshold of average RGB difference between template and source image. Default: +inf threshold (no rgbdiff)

    Returns: List of satisfied matched points in format [[point.x, point.y], angle, scale].
    """
    img_gray = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(rgbtemplate, cv2.COLOR_RGB2GRAY)
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    all_points = []
    if minmax == False:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                else:
                    rotated_template = rotate_image(scaled_template_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF_NORMED)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                else:
                    print("There's no such comparison method for template matching.")
                for pt in zip(*satisfied_points[::-1]):
                    all_points.append([pt, next_angle, actual_scale])
    else:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                else:
                    rotated_template = rotate_image(scaled_template_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                else:
                    print("There's no such comparison method for template matching.")
        if method == "TM_CCOEFF":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCOEFF_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_SQDIFF":
            all_points = sorted(all_points, key=lambda x: x[3])
        elif method == "TM_SQDIFF_NORMED":
            all_points = sorted(all_points, key=lambda x: x[3])
    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if ((abs(visited_point[0] - point[0]) < (width * scale / 100)) and (abs(visited_point[1] - point[1]) < (height * scale / 100))):
                        all_visited_points_not_close = False
                if all_visited_points_not_close == True:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            else:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        points_list = lone_points_list
    else:
        points_list = all_points
    if rgbdiff_thresh != float("inf"):
        print(">>>RGBDiff Filtering>>>")
        color_filtered_list = []
        template_channels = cv2.mean(rgbtemplate)
        template_channels = np.array([template_channels[0], template_channels[1], template_channels[2]])
        for point_info in points_list:
            point = point_info[0]
            cropped_img = rgbimage[point[1]:point[1]+height, point[0]:point[0]+width]
            cropped_channels = cv2.mean(cropped_img)
            cropped_channels = np.array([cropped_channels[0], cropped_channels[1], cropped_channels[2]])
            diff_observation = cropped_channels - template_channels
            total_diff = np.sum(np.absolute(diff_observation))
            print(total_diff)
            if total_diff < rgbdiff_thresh:
                color_filtered_list.append([point_info[0],point_info[1],point_info[2]])
        return color_filtered_list
    else:
        return points_list
    
def parse_points_list(points_list,width,height):
    """
    points_list: List of satisfied matched points in format [[point.x, point.y], angle, scale].
    
    Returns: List of satisfied matched points in format [[point.x, point.y], angle, scale].
    """
    centers_list = []
    box_list = []   
    for point_info in points_list:
        point = point_info[0]
        angle = point_info[1]
        scale = point_info[2]
        if angle >=0 and angle <= 5:
            print(f"matched point: {point}, angle: {angle}, scale: {scale}")
            centers_list.append([point, scale])
            box = patches.Rectangle((point[0], point[1]), width*scale/100, height*scale/100, color="green", alpha=0.50, label='Bounding box')
            box_list.append([point[0], point[1], point[0]+width*scale/100, point[1]+height*scale/100])
    return centers_list, box_list

def convert_box_to_mask(box_list, image_shape):
    """
    box_list: List of bounding box in format [x1, y1, x2, y2].
    image_shape: Shape of the image.
    
    Returns: Mask of bounding box.
    """
    res_masks = []
    empty_mask = torch.zeros((image_shape[0], image_shape[1]), dtype=torch.float32)
    if len(box_list) == 0:
        return (empty_mask,)
    for box in box_list:
        x1, y1,x2,y2 = map(int, box)
        empty_mask[y1:y2, x1:x2] = 255
    res_masks.append(empty_mask)
    
    return torch.cat(res_masks, dim=0)

class TemplateMatching:
    @classmethod
    def INPUT_TYPES(cls):
        method_list = ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"]
        return {
            "required": {
                "rgbimage": ('IMAGE', {}),
                "rgbtemplate": ('IMAGE', {}),
                "method": (method_list, {"default": "TM_CCOEFF_NORMED"}),
                "matched_thresh": ('FLOAT', {"default": 0.8}),
                "rot_range": ("STRING", {"default": "0,360"}),
                
                "rot_interval": ('INT', {"default": 180}),
                "scale_range": ("STRING", {"default": "50,200"}),
                "scale_interval": ('INT', {"default": 10}),
                "rm_redundant": ('BOOLEAN', {"default": True}),
                "minmax": ('BOOLEAN', {"default": True}),

            }
        }
    CATEGORY = "template_matching"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)



    def main(self, rgbimage, rgbtemplate, method, matched_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax):
        print(f"before: {rgbimage.shape}")
        rgbimage =  np.clip(255. * rgbimage.squeeze(0).cpu().numpy(), 0, 255).astype(np.uint8)
        print(f"after: {rgbimage.shape}")
        rgbtemplate =  np.clip(255. * rgbtemplate.squeeze(0).cpu().numpy(), 0, 255).astype(np.uint8)
        
        if ',' in rot_range:
            rot_range = [int(idx) for idx in rot_range.split(',')]
            if len(rot_range) != 2:
                logger.info("Invalid rot_range. Using default value.")
                rot_range = [0, 360]
        if ',' in scale_range:
            scale_range = [int(idx) for idx in scale_range.split(',')]
            if len(scale_range) != 2:
                logger.info("Invalid scale_range. Using default value.")
                scale_range = [50, 200]
        points_list = invariant_match_template(rgbimage, rgbtemplate, method, matched_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax)
        mask = torch.zeros((rgbimage.shape[0], rgbimage.shape[1]), dtype=torch.float32)
        if len(points_list) == 0:
            return (mask,)
        centers_list, box_list = parse_points_list(points_list, rgbtemplate.shape[1], rgbtemplate.shape[0])
        print(f"box_list: {box_list}")
        mask = convert_box_to_mask(box_list, [rgbimage.shape[0], rgbimage.shape[1]])
        return (mask,)


class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]

    FUNCTION = "main"
    CATEGORY = "template_matching"

    def main(self, mask):
        return (torch.all(mask == 0).int().item(), )