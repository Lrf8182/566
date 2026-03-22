import numpy as np

class SimpleRouter:
    def __init__(
        self,
        conf_easy=0.35,
        mean_conf_thr=0.45,
        num_boxes_thr=18,
        small_obj_ratio_thr=0.55,
        min_area_ratio=0.01,
    ):
        self.conf_easy = conf_easy
        self.mean_conf_thr = mean_conf_thr
        self.num_boxes_thr = num_boxes_thr
        self.small_obj_ratio_thr = small_obj_ratio_thr
        self.min_area_ratio = min_area_ratio

    def decide(self, boxes_xyxy, scores, image_shape):
        """
        返回:
            escalate: bool
            stats: dict
        """
        h, w = image_shape[:2]
        img_area = h * w

        if len(scores) == 0:
            return True, {
                "num_boxes": 0,
                "mean_conf": 0.0,
                "small_ratio": 1.0,
                "reason": "no_detection"
            }

        scores = np.array(scores, dtype=np.float32)
        boxes = np.array(boxes_xyxy, dtype=np.float32)

        wh = boxes[:, 2:4] - boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        area_ratio = areas / max(img_area, 1)

        small_ratio = float((area_ratio < self.min_area_ratio).mean())
        mean_conf = float(scores.mean())
        num_boxes = int(len(scores))

        hard_case = (
            mean_conf < self.mean_conf_thr
            or num_boxes > self.num_boxes_thr
            or small_ratio > self.small_obj_ratio_thr
        )

        reason = []
        if mean_conf < self.mean_conf_thr:
            reason.append("low_conf")
        if num_boxes > self.num_boxes_thr:
            reason.append("crowded")
        if small_ratio > self.small_obj_ratio_thr:
            reason.append("many_small_objects")

        return hard_case, {
            "num_boxes": num_boxes,
            "mean_conf": mean_conf,
            "small_ratio": small_ratio,
            "reason": "+".join(reason) if reason else "easy"
        }