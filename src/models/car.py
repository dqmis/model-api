from pydantic import BaseModel


class Input(BaseModel):
    fuel: str
    seller_type: str
    transmission: str
    name: str
    km_driven: float
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float
    year: int
