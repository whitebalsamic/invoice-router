from typing import List, Optional

from pydantic import BaseModel, Field

from .enums import JobStatus


class JobProgress(BaseModel):
    job_id: str
    total: int = 0
    done: int = 0
    failed: int = 0
    current_item: Optional[str] = None
    accuracy: float = 0.0
    status: JobStatus = JobStatus.PENDING
    errors: List[str] = Field(default_factory=list)
