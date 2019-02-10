from typing import Any, Dict, Optional


class ImageProcessingError(Exception):
    default_message: str = "Unable to process image"

    def __init__(
            self,
            filename: str = '',
            message: Optional[str] = None,
            extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.filename: str = filename
        self.message: str = message or self.default_message
        self.extra_info: Optional[Dict[str, Any]] = extra_info
        super().__init__()

    def __str__(self) -> str:
        return self.get_message(with_filename=True, with_extra_info=True)

    def get_message(
            self,
            *,
            with_filename: bool = False,
            with_extra_info: bool = True,
    ) -> str:
        add_filename = (self.filename and with_filename)
        from_file = f' from file: {self.filename}' if add_filename else ''
        extra_info = self.extra_info or {}
        extra = ', '.join(f'{k} = {v}' for (k, v) in extra_info.items())
        extra_suffix = f' ({extra})' if extra and with_extra_info else ''
        return f'{self.message}{from_file}{extra_suffix}'


class ImageLoadingError(ImageProcessingError, IOError):
    default_message = "Unable to load image"


class ImageAnalyzingError(ImageProcessingError, ValueError):
    default_message = "Failed to analyze image"


class DialsNotFoundError(ImageAnalyzingError):
    default_message = "Dials not found"


class DialAngleDeterminingError(ImageAnalyzingError):
    default_message = "Cannot determine angle of a dial"


class NeedleContoursNotFoundError(ImageAnalyzingError):
    default_message = "Cannot find needle contours of a dial"
