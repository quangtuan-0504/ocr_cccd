class CardDetectionError(Exception):
    """Exception khi không phát hiện được thẻ CCCD"""
    pass

class BadCaptureError(Exception):
    """Exception khi ảnh chụp không đạt chất lượng"""
    pass

class BadBoxError(Exception):
    """Exception khi bounding box không hợp lệ"""
    pass

class ModelNotLoadedError(Exception):
    """Exception khi model chưa được load"""
    pass