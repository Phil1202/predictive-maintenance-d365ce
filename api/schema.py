from pydantic import BaseModel, Field, conlist

class PredictionInput(BaseModel):
    data: conlist(float, min_length=12, max_length=12) = Field(
        ..., description="Exakt 12 numerische Features in der richtigen Reihenfolge"
    )
