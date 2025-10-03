import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class VehicleUpdate(Base):
    __tablename__ = "vehicle_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    trip_id = Column(String, nullable=True)
    vehicle_id = Column(String, nullable=True)
    last_stop_id = Column(String, nullable=True)
    delay_seconds = Column(Integer, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "transit.db")
    engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
    Base.metadata.create_all(engine)
    print(f"Tables created in {os.path.abspath(db_path)}")
