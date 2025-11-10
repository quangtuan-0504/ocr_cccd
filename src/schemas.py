"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

class ResponseMessage:
    # Success messages
    SUCCESS = "Success"

    # Error messages
    # Generic errors
    INPUT_ERROR = "Input image error: {}"
    PROCESS_ERROR = "Error processing image: {}"
    BAD_CAPTURE = "Bad Capture issue: {}"
    CARD_DETECT_ERROR = "Card Detect issue: {}"
    BAD_BOX = "Bad Box issue: {}"
    TEXT_REGION_DET_ERROR = "Text Region Detect issue: {}"
    TEXT_EXTRACT_ERROR= "Text Extract issue: {}"

    @staticmethod
    def input_error(err: str) -> str:
        return ResponseMessage.INPUT_ERROR.format(err)
    @staticmethod
    def bad_capture(err: str) -> str:
        return ResponseMessage.BAD_CAPTURE.format(err)
    @staticmethod
    def card_detect_error(err: str) -> str:
        return ResponseMessage.CARD_DETECT_ERROR.format(err)
    @staticmethod
    def bad_box(err: str) -> str:
        return ResponseMessage.BAD_BOX.format(err)
    @staticmethod
    def text_region_det_error(err: str) -> str:
        return ResponseMessage.TEXT_REGION_DET_ERROR.format(err)
    @staticmethod
    def text_extract_error(err: str) -> str:
        return ResponseMessage.TEXT_EXTRACT_ERROR.format(err)
    @staticmethod
    def process_img_error(err: str) -> str:
        return ResponseMessage.PROCESS_ERROR.format(err)
    @staticmethod
    def error_occurred(error: Exception) -> str:
        return f"An unexpected error occurred: {str(error)}"

@dataclass
class BaseResponse:
    data: List|None = None
    errorCode: int = 200
    errorMessage: str = ResponseMessage.SUCCESS

cccd_front = {
    "id": "",
    "id_prob": "",
    "name": "",
    "name_prob": "",
    "dob": "",
    "dob_prob": "",
    "sex": "",
    "sex_prob": "",
    "nationality": "",
    "nationality_prob": "",
    "home": "",
    "home_prob": "",
    "address": "",
    "address_prob": "",
    "doe": "",
    "doe_prob": "",
    "type": "",
    "type_new": "N/A",
    "overall_score": "",
    "custAddress": {
        "province": "",
        "district": "",
        "ward": "",
        "street": ""
    }
}
cccd_back = {
    "features": "",
    "features_prob": "",
    "issue_date": "",
    "issue_date_prob": "",
    "type": "",
    "type_new": "N/A",
    "overall_score": "",
    "mrz": [
        "N/A",
        "N/A",
        "N/A"
    ],
    "mrz_details": {
        "id": "N/A",
        "name": "N/A",
        "doe": "N/A",
        "dob": "N/A",
        "nationality": "N/A",
        "sex": "N/A"
    }
}
    
cc_front = {
    "id": "",
    "id_prob": "",
    "name": "",
    "name_prob": "",
    "dob": "",
    "dob_prob": "",
    "sex": "",
    "sex_prob": "",
    "nationality": "",
    "nationality_prob": "",
    "type": "",
    "type_new": "N/A",
    "overall_score": "",
}
cc_back = {
    "address": "",
    "address_prob": "",
    "home": "",
    "home_prob": "",
    "issue_date": "",
    "issue_date_prob": "",
    "doe": "",
    "doe_prob": "",
    "features": "N/A",
    "features_prob": "N/A",
    "issue_loc_prob": "",
    "issue_loc": "BỘ CÔNG AN",
    "type": "",
    "type_new": "N/A",
    "overall_score": "",
    "custAddress": {
        "province": "",
        "district": "",
        "ward": "",
        "street": ""
    },
    "mrz": [
        "N/A",
        "N/A",
        "N/A"
    ],
    "mrz_details": {
        "id": "N/A",
        "name": "N/A",
        "doe": "N/A",
        "dob": "N/A",
        "nationality": "N/A",
        "sex": "N/A"
    }
}


cccd_front_fields = {
    'id': 'id',
    'name':'name',
    'date_of_birth':'dob',
    'sex':'sex',
    'nation':'nationality',
    'place_of_origin':'home',
    'place_of_residence':'address',
    'date_of_expiry':'doe'
}
cccd_back_fields = {
    'personal_iden':'features',
    'date_of_issue':'issue_date',
    'mrz': 'mrz'
}
cc_front_fields = {
    'id' : 'id',
    'name': 'name',
    'date_of_birth' :'dob',
    'sex' : 'sex',
    'nation' :'nationality'
}
cc_back_fields = {
    'place_of_residence' :'address',
    'place_of_birth' :'home',
    'date_of_issue' :'issue_date',
    'date_of_expiry' :'doe',
    'mrz' : 'mrz'
}

format_res = {    
    "id": "",
 
    "id_prob": "",
 
    "name": "",
 
    "name_prob": "",
 
    "dob": "",
 
    "dob_prob": "",
 
    "sex": "",
 
    "sex_prob": "",
 
    "nationality": "",
 
    "nationality_prob": "",
 
    "home": "",
 
    "home_prob": "",
 
    "address": "",
 
    "address_prob": "",
 
    "doe": "",
 
    "doe_prob": "",
 
    "religion": "",
 
    "religion_prob": "",
 
    "ethnicity": "",
 
    "ethnicity_prob": "",
 
    "features": "",
 
    "features_prob": "",
 
    "issue_date": "",
 
    "issue_date_prob": "",
 
    "issue_loc": "",
 
    "issue_loc_prob": "",
 
    "back_type": "N/A",
 
    "overall_score": "",
 
    "type": "",
 
    "type_new": "",
 
    "mrz": "N/A",
 
    "mrz_prob": "N/A",
 
    "mrz_details": {
 
        "id": "N/A",
 
        "name": "N/A",
 
        "doe": "N/A",
 
        "dob": "N/A",
 
        "nationality": "N/A",
 
        "sex": "N/A"
 
    },
 
    "custAddress": {
 
        "province": "",
 
        "district": "",
 
        "ward": "",
 
        "street": ""
 
    }
}
cccd_map = {
    'id': 'id',
    'name':'name',
    'date_of_birth':'dob',
    'sex':'sex',
    'nation':'nationality',
    'place_of_origin':'home',
    'place_of_residence':'address',
    'date_of_expiry':'doe',
    'personal_iden':'features',
    'date_of_issue':'issue_date',
    'mrz': 'mrz'
}
cc_map = {
    'id' : 'id',
    'name': 'name',
    'date_of_birth' :'dob',
    'sex' : 'sex',
    'nation' :'nationality',
    'place_of_residence' :'address',
    'place_of_birth' :'home',
    'date_of_issue' :'issue_date',
    'date_of_expiry' :'doe',
    'mrz' : 'mrz'
}
 