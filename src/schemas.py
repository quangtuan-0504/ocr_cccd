"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class OCRResponse(BaseModel):
    """Response model for OCR API endpoints."""
    status: str = Field(..., description="Status of the OCR process (success or error)")
    message: Optional[str] = Field(None, description="Error message if status is error")
    card_info: Optional[Dict[str, Any]] = Field(None, description="Information about the detected card")
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted text data from the card")

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
 