from pydantic import BaseModel
from typing import Optional

class ReceiptExtraction(BaseModel):
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    transaction_date: Optional[str] = None
    total: Optional[float] = None