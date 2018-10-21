from typing import Dict, Optional

import cv2
import numpy

from ._params import Params as _Params
from ._types import Image, TemplateMatchResult
from ._utils import convert_to_hls, crop_rect, match_template


class ImageFile:
    def __init__(
            self,
            filename: str,
            params: _Params,
            bgr_image: Optional[Image] = None,
    ) -> None:
        self.filename = filename
        self.params = params
        self.bgr_image = bgr_image

    def get_dials_hls(self) -> Image:
        hls_image = self.get_hls_image()
        match_result = self._find_dials(hls_image)
        dials_hls = crop_rect(hls_image, match_result.rect)
        return dials_hls

    def get_hls_image(self) -> Image:
        bgr_image = self.get_bgr_image()
        hls_image = convert_to_hls(bgr_image, self.params.hue_shift)
        return hls_image

    def get_bgr_image_t(self) -> Image:
        bgr_image = self.get_bgr_image()
        hls_image = convert_to_hls(bgr_image, self.params.hue_shift)
        dials = self._find_dials(hls_image)
        tl = dials.rect.top_left
        m = numpy.array([
            [1, 0, 30 - tl[0]],
            [0, 1, 116 - tl[1]]
        ], dtype=numpy.float32)
        (h, w) = bgr_image.shape[0:2]
        return cv2.warpAffine(bgr_image, m, (w, h))

    def get_bgr_image(self) -> Image:
        if self.bgr_image is not None:
            return self.bgr_image
        img = cv2.imread(self.filename)
        if img is None:
            raise Exception(
                "Unable to read image file: {}".format(self.filename))
        return self._crop_meter(img)

    def _crop_meter(self, img: Image) -> Image:
        return crop_rect(img, self.params.meter_rect)

    def _find_dials(self, img_hls: Image) -> TemplateMatchResult:
        template = _get_dials_template(self.params)
        lightness = cv2.split(img_hls)[1]
        match_result = match_template(lightness, template)

        if match_result.max_val < self.params.dials_match_threshold:
            raise ValueError('Dials not found from {} (match val = {})'.format(
                self.filename, match_result.max_val))

        return match_result


_dials_template_map: Dict[int, Image] = {}


def _get_dials_template(params: _Params) -> Image:
    dials_template = _dials_template_map.get(id(params))
    if dials_template is None:
        dials_template = cv2.imread(params.dials_file, cv2.IMREAD_GRAYSCALE)
        if dials_template is None:
            raise IOError(
                "Cannot read dials template: {}".format(params.dials_file))
        _dials_template_map[id(params)] = dials_template
    assert dials_template.shape == params.dials_template_size
    return dials_template
