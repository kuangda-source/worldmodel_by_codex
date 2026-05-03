from __future__ import annotations

from ..schemas import Vehicle
from ..storage import list_vehicles, put_vehicle


def get_vehicles() -> list[Vehicle]:
    return list_vehicles()


def save_vehicle(vehicle: Vehicle) -> Vehicle:
    return put_vehicle(vehicle)
