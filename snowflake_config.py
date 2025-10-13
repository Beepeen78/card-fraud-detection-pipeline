"""
Snowflake configuration and connection utilities for the fraud detection pipeline.
"""
import os
import pandas as pd
from typing import Optional, Dict, Any
import logging

# Try to import Snowflake modules, but don't crash if they're not available
try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    logging.warning("Snowflake packages not available. Snowflake functionality will be disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeConfig:
    """Configuration class for Snowflake connection parameters."""
    
    def __init__(self):
        # Snowflake connection parameters from environment variables
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.database = os.getenv("SNOWFLAKE_DATABASE", "FRAUD_DETECTION")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", "PRODUCTION")
        self.role = os.getenv("SNOWFLAKE_ROLE", "PUBLIC")
        
        # Table configurations
        self.transactions_table = "TRANSACTIONS_SCORED"
        self.metrics_table = "METRICS_DAILY"
        
    def validate_config(self) -> bool:
        """Validate that required Snowflake configuration is present."""
        required_fields = [self.account, self.user, self.password]
        if not all(required_fields):
            missing = [field for field, value in zip(
                ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"], 
                required_fields
            ) if not value]
            logger.error(f"Missing required Snowflake configuration: {missing}")
            return False
        return True

class SnowflakeConnector:
    """Snowflake connection and data operations manager."""
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        self.config = config or SnowflakeConfig()
        self.connection = None
        
    def connect(self) -> bool:
        """Establish connection to Snowflake."""
        if not SNOWFLAKE_AVAILABLE:
            logger.error("Snowflake packages not available")
            return False
            
        if not self.config.validate_config():
            return False
            
        try:
            self.connection = snowflake.connector.connect(
                account=self.config.account,
                user=self.config.user,
                password=self.config.password,
                warehouse=self.config.warehouse,
                database=self.config.database,
                schema=self.config.schema,
                role=self.config.role
            )
            logger.info("Successfully connected to Snowflake")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            return False
    
    def disconnect(self):
        """Close Snowflake connection."""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Snowflake")
    
    def upload_dataframe(self, df: pd.DataFrame, table_name: str, 
                        if_exists: str = "append") -> bool:
        """Upload pandas DataFrame to Snowflake table."""
        if not SNOWFLAKE_AVAILABLE:
            logger.error("Snowflake packages not available")
            return False
            
        if not self.connection:
            logger.error("No active Snowflake connection")
            return False
            
        try:
            # Ensure table exists
            self._create_table_if_not_exists(table_name, df)
            
            # Upload data
            success, nchunks, nrows, _ = write_pandas(
                conn=self.connection,
                df=df,
                table_name=table_name,
                database=self.config.database,
                schema=self.config.schema,
                auto_create_table=False,
                overwrite=(if_exists == "replace")
            )
            
            if success:
                logger.info(f"Successfully uploaded {nrows} rows to {table_name}")
                return True
            else:
                logger.error(f"Failed to upload data to {table_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading to {table_name}: {e}")
            return False
    
    def _create_table_if_not_exists(self, table_name: str, df: pd.DataFrame):
        """Create table with appropriate schema if it doesn't exist."""
        try:
            cursor = self.connection.cursor()
            
            # Define table schemas based on table name
            if table_name.upper() == self.config.transactions_table:
                schema_sql = """
                CREATE TABLE IF NOT EXISTS {}.{}.{} (
                    TRANSACTION_ID VARCHAR(255),
                    CUSTOMER_ID VARCHAR(255),
                    AMOUNT FLOAT,
                    FRAUD_PROBABILITY FLOAT,
                    FRAUD_PREDICTION INTEGER,
                    IS_FRAUD INTEGER,
                    SCORE_TIME TIMESTAMP_NTZ,
                    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
                """.format(self.config.database, self.config.schema, table_name)
                
            elif table_name.upper() == self.config.metrics_table:
                schema_sql = """
                CREATE TABLE IF NOT EXISTS {}.{}.{} (
                    DATE DATE,
                    TRANSACTIONS INTEGER,
                    FLAGGED INTEGER,
                    AVG_RISK FLOAT,
                    TOTAL_AMOUNT FLOAT,
                    ACTUAL_FRAUD INTEGER,
                    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
                """.format(self.config.database, self.config.schema, table_name)
            else:
                logger.warning(f"No predefined schema for table {table_name}")
                return
            
            cursor.execute(schema_sql)
            cursor.close()
            logger.info(f"Table {table_name} schema ensured")
            
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
    
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute a SQL query and return results as DataFrame."""
        if not self.connection:
            logger.error("No active Snowflake connection")
            return None
            
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            
            return pd.DataFrame(results, columns=columns)
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None

def upload_to_snowflake(df: pd.DataFrame, table_name: str, 
                       config: Optional[SnowflakeConfig] = None) -> str:
    """Convenience function to upload DataFrame to Snowflake."""
    connector = SnowflakeConnector(config)
    
    if not connector.connect():
        return "Failed to connect to Snowflake"
    
    try:
        success = connector.upload_dataframe(df, table_name)
        if success:
            return f"Successfully uploaded {len(df)} rows to {table_name}"
        else:
            return f"Failed to upload data to {table_name}"
    finally:
        connector.disconnect()

# Example usage and configuration setup
def setup_snowflake_env_example():
    """Example of how to set up Snowflake environment variables."""
    example_env = """
# Add these to your environment or .env file:
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=FRAUD_DETECTION
SNOWFLAKE_SCHEMA=PRODUCTION
SNOWFLAKE_ROLE=PUBLIC
"""
    print("Snowflake Environment Setup Example:")
    print(example_env)
