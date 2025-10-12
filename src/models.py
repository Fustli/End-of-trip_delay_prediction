"""
Database setup and management for BKK Real-time Vehicle Data Scraper.
Production-ready with proper error handling and migrations.
"""

import os
import logging
import datetime
import pytz
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, event
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import StaticPool

# Import configuration
import config

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

Base = declarative_base()

class VehicleUpdate(Base):
    """
    Production database model for storing vehicle position and delay data.
    Includes data quality flags for robust GNN training.
    """
    __tablename__ = "vehicle_updates"

    # Primary key and core fields
    id = Column(Integer, primary_key=True, autoincrement=True, comment="Unique record identifier")
    timestamp = Column(DateTime, nullable=False, index=True, comment="Data collection timestamp")
    trip_id = Column(String(50), nullable=True, index=True, comment="GTFS trip identifier")
    vehicle_id = Column(String(50), nullable=True, index=True, comment="Vehicle identifier")
    last_stop_id = Column(String(20), nullable=True, comment="Last visited stop ID")
    
    # Delay information
    delay_seconds = Column(Integer, nullable=True, comment="Calculated delay in seconds (NULL if not applicable)")
    
    # Position data
    latitude = Column(Float, nullable=True, comment="Vehicle latitude")
    longitude = Column(Float, nullable=True, comment="Vehicle longitude")
    
    # Data Quality Flags (for GNN training filtering)
    has_position = Column(Boolean, default=False, nullable=False, comment="Has valid GPS coordinates")
    has_stop_info = Column(Boolean, default=False, nullable=False, comment="Has valid stop information")
    is_endpoint = Column(Boolean, default=False, nullable=False, comment="Vehicle at route endpoint")
    delay_calculated = Column(Boolean, default=False, nullable=False, comment="Delay successfully calculated")
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(pytz.timezone('Europe/Budapest')), 
                       nullable=False, index=True, comment="Record creation timestamp")

    def __repr__(self):
        return f"<VehicleUpdate(trip={self.trip_id}, vehicle={self.vehicle_id}, delay={self.delay_seconds}s)>"

    def to_dict(self):
        """Convert model to dictionary for easy serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'trip_id': self.trip_id,
            'vehicle_id': self.vehicle_id,
            'last_stop_id': self.last_stop_id,
            'delay_seconds': self.delay_seconds,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'has_position': self.has_position,
            'has_stop_info': self.has_stop_info,
            'is_endpoint': self.is_endpoint,
            'delay_calculated': self.delay_calculated,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# =============================================================================
# DATABASE ENGINE & SESSION MANAGEMENT
# =============================================================================

_engine = None
_Session = None

def get_engine():
    """Create and return database engine with production settings."""
    global _engine
    if _engine is None:
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
            
            _engine = create_engine(
                config.DATABASE_URL,
                echo=config.DEBUG,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                    "isolation_level": None  # SQLite autocommit mode
                }
            )
            
            # Add connection event handlers
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
                cursor.execute("PRAGMA synchronous=NORMAL")  # Good balance of safety/performance
                cursor.close()
                
            logger.info(f"✅ Database engine created for: {config.DB_PATH}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create database engine: {e}")
            raise
    
    return _engine

def get_session():
    """Create and return database Session class."""
    global _Session
    if _Session is None:
        engine = get_engine()
        _Session = sessionmaker(bind=engine)
        logger.info("✅ Session factory created")
    
    return _Session

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    Session = get_session()
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Session rollback due to error: {e}")
        raise
    finally:
        session.close()

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def init_database():
    """
    Initialize database tables. Handles existing tables gracefully.
    """
    try:
        engine = get_engine()
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        logger.info(f"✅ Database initialized successfully at: {os.path.abspath(config.DB_PATH)}")
        logger.info(f"✅ Table '{VehicleUpdate.__tablename__}' ready with {len(VehicleUpdate.__table__.columns)} columns")
        
        # Log detailed table structure
        columns_info = []
        for col in VehicleUpdate.__table__.columns:
            index_info = " [INDEXED]" if col.index else ""
            columns_info.append(f"{col.name} ({col.type}){index_info}")
        
        logger.info(f"📊 Table structure: {', '.join(columns_info)}")
        
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during database initialization: {e}")
        return False

def check_database_health():
    """
    Check database connectivity and basic functionality.
    """
    try:
        with session_scope() as session:
            # Test basic queries
            record_count = session.query(VehicleUpdate).count()
            recent_records = session.query(VehicleUpdate).order_by(VehicleUpdate.id.desc()).limit(5).all()
            
            logger.info(f"📈 Database health check passed - {record_count} total records")
            logger.info(f"📈 Most recent record ID: {recent_records[0].id if recent_records else 'None'}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return False

def get_database_stats():
    """
    Get comprehensive database statistics.
    """
    try:
        with session_scope() as session:
            stats = {
                'total_records': session.query(VehicleUpdate).count(),
                'records_with_delays': session.query(VehicleUpdate).filter(VehicleUpdate.delay_calculated == True).count(),
                'records_at_endpoints': session.query(VehicleUpdate).filter(VehicleUpdate.is_endpoint == True).count(),
                'records_with_positions': session.query(VehicleUpdate).filter(VehicleUpdate.has_position == True).count(),
                'records_with_stop_info': session.query(VehicleUpdate).filter(VehicleUpdate.has_stop_info == True).count(),
                'earliest_record': session.query(VehicleUpdate.timestamp).order_by(VehicleUpdate.timestamp.asc()).first(),
                'latest_record': session.query(VehicleUpdate.timestamp).order_by(VehicleUpdate.timestamp.desc()).first(),
            }
            
            # Calculate percentages
            total = stats['total_records']
            if total > 0:
                stats['delay_percentage'] = round(stats['records_with_delays'] / total * 100, 2)
                stats['position_percentage'] = round(stats['records_with_positions'] / total * 100, 2)
                stats['stop_info_percentage'] = round(stats['records_with_stop_info'] / total * 100, 2)
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}

def cleanup_old_records(days_to_keep=30):
    """
    Clean up records older than specified days to manage database size.
    """
    try:
        cutoff_date = datetime.datetime.now(pytz.timezone('Europe/Budapest')) - datetime.timedelta(days=days_to_keep)
        
        with session_scope() as session:
            deleted_count = session.query(VehicleUpdate).filter(
                VehicleUpdate.created_at < cutoff_date
            ).delete()
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} records older than {days_to_keep} days")
            
            return deleted_count
            
    except Exception as e:
        logger.error(f"Failed to clean up old records: {e}")
        return 0

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize database when run directly
    success = init_database()
    
    if success:
        # Run comprehensive health check
        health_ok = check_database_health()
        
        if health_ok:
            stats = get_database_stats()
            print(f"🎉 Database ready for production!")
            print(f"📊 Current stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # Suggest cleanup if database is large
            if stats.get('total_records', 0) > 1000000:
                print("💡 Consider running cleanup_old_records() to manage database size")
        else:
            print("⚠️  Database created but health check failed")
    else:
        print("❌ Database initialization failed")
        exit(1)