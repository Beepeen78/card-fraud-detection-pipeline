"""
Python script to set up Snowflake database, schema, and tables for the fraud detection pipeline.
This script can be run to initialize your Snowflake environment.
"""
import os
import sys
from pathlib import Path
from snowflake_config import SnowflakeConnector, SnowflakeConfig

def setup_snowflake():
    """Set up Snowflake database, schema, and tables."""
    
    # Check if Snowflake is configured
    config = SnowflakeConfig()
    if not config.validate_config():
        print("❌ Snowflake configuration is missing.")
        print("Please set the following environment variables:")
        print("- SNOWFLAKE_ACCOUNT")
        print("- SNOWFLAKE_USER") 
        print("- SNOWFLAKE_PASSWORD")
        print("- SNOWFLAKE_WAREHOUSE (optional, defaults to COMPUTE_WH)")
        print("- SNOWFLAKE_DATABASE (optional, defaults to FRAUD_DETECTION)")
        print("- SNOWFLAKE_SCHEMA (optional, defaults to PRODUCTION)")
        print("- SNOWFLAKE_ROLE (optional, defaults to PUBLIC)")
        return False
    
    # Read SQL setup file
    sql_file = Path("setup_snowflake.sql")
    if not sql_file.exists():
        print(f"❌ SQL setup file not found: {sql_file}")
        return False
    
    with open(sql_file, 'r') as f:
        sql_commands = f.read()
    
    # Split SQL commands by semicolon and filter out empty ones
    commands = [cmd.strip() for cmd in sql_commands.split(';') if cmd.strip()]
    
    # Connect to Snowflake and execute commands
    connector = SnowflakeConnector(config)
    
    if not connector.connect():
        print("❌ Failed to connect to Snowflake")
        return False
    
    try:
        print("🔗 Connected to Snowflake successfully")
        print(f"📍 Database: {config.database}")
        print(f"📍 Schema: {config.schema}")
        
        cursor = connector.connection.cursor()
        
        for i, command in enumerate(commands, 1):
            if command.startswith('--') or not command:
                continue
                
            print(f"⚡ Executing command {i}/{len(commands)}...")
            try:
                cursor.execute(command)
                print(f"✅ Command {i} executed successfully")
            except Exception as e:
                print(f"⚠️  Command {i} failed: {e}")
                # Continue with other commands
        
        cursor.close()
        
        # Verify tables were created
        print("\n🔍 Verifying table creation...")
        cursor = connector.connection.cursor()
        
        # Check transactions table
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = '{config.schema}' 
            AND TABLE_NAME = '{config.transactions_table}'
        """)
        transactions_exists = cursor.fetchone()[0] > 0
        
        # Check metrics table
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = '{config.schema}' 
            AND TABLE_NAME = '{config.metrics_table}'
        """)
        metrics_exists = cursor.fetchone()[0] > 0
        
        cursor.close()
        
        if transactions_exists and metrics_exists:
            print("✅ All tables created successfully!")
            print(f"✅ {config.transactions_table} table exists")
            print(f"✅ {config.metrics_table} table exists")
            print("\n🎉 Snowflake setup completed successfully!")
            return True
        else:
            print("❌ Some tables were not created successfully")
            return False
            
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False
    finally:
        connector.disconnect()

def test_connection():
    """Test Snowflake connection and configuration."""
    config = SnowflakeConfig()
    
    print("Testing Snowflake connection...")
    print(f"Account: {config.account}")
    print(f"User: {config.user}")
    print(f"Warehouse: {config.warehouse}")
    print(f"Database: {config.database}")
    print(f"Schema: {config.schema}")
    print(f"Role: {config.role}")
    
    if not config.validate_config():
        print("Configuration validation failed")
        return False
    
    connector = SnowflakeConnector(config)
    
    if connector.connect():
        print("Connection test successful!")
        connector.disconnect()
        return True
    else:
        print("Connection test failed")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connection()
    else:
        print("Setting up Snowflake for Fraud Detection Pipeline")
        print("=" * 50)
        success = setup_snowflake()
        
        if success:
            print("\nNext steps:")
            print("1. Your Snowflake environment is ready!")
            print("2. You can now run the Streamlit app and export to Snowflake")
            print("3. Check the created views for analytics:")
            print("   - HIGH_RISK_TRANSACTIONS")
            print("   - DAILY_FRAUD_SUMMARY")
        else:
            print("\nSetup failed. Please check your configuration and try again.")
            print("Run 'python setup_snowflake.py test' to test your connection.")
