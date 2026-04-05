from pydantic import BaseModel, Field
from typing import Annotated

class UserInput(BaseModel):
  magnitude : Annotated[float,Field(
    ..., description="Richter scale magnitude (required)",
    ge = -1.0, le = 10.0, examples=[7.1,6.0] 
  )]
  depth : Annotated[float,Field(
    ..., description="Depth of earthquake in kilometers (required)",
    ge = 0, le = 700, examples=[622.0,41.0]
  )]
  cdi : Annotated[float,Field(
    ...,description="Community Decimal Intensity (required)",
    ge = 0, le = 10, examples=[3.0,9.0]
  )]
  mmi : Annotated[float,Field(
    ...,description="Modified Mercalli Intensity (required)",
    ge = 1.0, le = 12.0, examples=[6.5,7.0]
  )]
  sig : Annotated[int,Field(
    ...,description="Significance parameter (required)",
    ge = -200, le = 800, examples=[120,-85]
  )]