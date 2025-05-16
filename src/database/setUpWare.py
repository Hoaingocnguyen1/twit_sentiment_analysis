from baseDatabase import Database
from dotenv import load_dotenv
import os
import tempfile
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)

# 1. Metadata dict cho tweet_staging
#    Lưu dữ liệu đã clean, chờ xử lý NLP và chuyển sang product
tweet_staging = {
    'tweet_id': 'BIGSERIAL PRIMARY KEY',  # Thay SERIAL bằng BIGSERIAL
    'topic': 'VARCHAR(255) NOT NULL',
    'content': 'TEXT NOT NULL',
    'crawl_at': 'TIMESTAMP NOT NULL',
    'created_at': 'TIMESTAMP NOT NULL',
    'predicted_sentiment': 'INT DEFAULT NULL',
    'updated_at': 'TIMESTAMP NOT NULL DEFAULT NOW()',
    'moved_to_product': 'BOOLEAN NOT NULL DEFAULT FALSE',
    'moved_at': 'TIMESTAMP DEFAULT NULL',
    'processed_done': 'BOOLEAN NOT NULL DEFAULT FALSE',
    'processed_at': 'TIMESTAMP DEFAULT NULL'
}

# 2. Metadata dict cho tweet_product
#    Lưu kết quả cuối cùng, chỉ giữ cột cần thiết cho downstream
tweet_product = {
    'staging_tweet_id': 'BIGINT PRIMARY KEY REFERENCES "TWEET_STAGING"(tweet_id)',  # Thay INT bằng BIGINT
    'topic': 'VARCHAR(255) NOT NULL',
    'predicted_sentiment': 'INT NOT NULL',
    'updated_at': 'TIMESTAMP NOT NULL DEFAULT NOW()',
    'created_at': 'TIMESTAMP NOT NULL',
    'processed_at': 'TIMESTAMP NOT NULL DEFAULT NOW()'
}

if __name__ == '__main__':
    load_dotenv()  # Load environment variables from .env file
    logging.info("Loading environment variables...")
    user = os.getenv('PG_USER')
    host = os.getenv('PG_HOST')
    port = os.getenv('PG_PORT')
    password = os.getenv('PG_PASSWORD')
    dbname = os.getenv('PG_DB1')
    sslmode = os.getenv('PG_SSLMODE')

    # Initialize database connection
    try:
        db = Database(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=sslmode
        )
        logging.info("Warehouse connection is completed.")
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        exit(1)

# Create TWEET_STAGING table
    db.create_table('TWEET_STAGING', columns=tweet_staging)
    logging.info("TWEET_STAGING table created.")

    # Create indexes for TWEET_STAGING
    db.create_index('TWEET_STAGING', 'idx_staging_crawl_at', ['crawl_at'])
    db.create_index('TWEET_STAGING', 'idx_staging_processed', ['processed_done', 'processed_at'])
    logging.info("Indexes for TWEET_STAGING created.")

    # Create trigger for updated_at in TWEET_STAGING
    db.create_trigger(
        table_name='TWEET_STAGING',
        trigger_name='tg_updated_at_staging',
        function_name='trg_set_updated_at_staging',
        function_body='BEGIN NEW.updated_at = NOW(); RETURN NEW; END;',
        timing='BEFORE',
        event='UPDATE',
        for_each='ROW'
    )
    logging.info("Trigger tg_updated_at_staging created.")

    # Create TWEET_PRODUCT table
    db.create_table('TWEET_PRODUCT', columns=tweet_product)
    logging.info("TWEET_PRODUCT table created.")

    # Create indexes for TWEET_PRODUCT
    db.create_index('TWEET_PRODUCT', 'idx_product_processed_at', ['processed_at'])
    db.create_index('TWEET_PRODUCT', 'idx_product_sentiment', ['predicted_sentiment'])
    logging.info("Indexes for TWEET_PRODUCT created.")

    # Create trigger to update TWEET_STAGING after update in TWEET_PRODUCT
    trigger_body_update_staging_from_product = '''
    BEGIN
        UPDATE TWEET_STAGING
        SET 
            processed_done = TRUE,
            processed_at = NEW.processed_at,
            moved_to_product = TRUE,
            moved_at = NEW.processed_at
        WHERE tweet_id = NEW.staging_tweet_id;
        RETURN NEW;
    END;
    '''
    db.create_trigger(
        table_name='TWEET_PRODUCT',
        trigger_name='tg_update_staging_on_product_update',
        function_name='trg_update_staging_on_product_update',
        function_body=trigger_body_update_staging_from_product,
        timing='AFTER',
        event='UPDATE',
        for_each='ROW'
    )
    logging.info("Trigger tg_update_staging_on_product_update created.")

    # Create trigger for updated_at in TWEET_PRODUCT
    db.create_trigger(
        table_name='TWEET_PRODUCT',
        trigger_name='tg_set_updated_at_product',
        function_name='trg_set_updated_at_product',
        function_body='BEGIN NEW.updated_at = NOW(); RETURN NEW; END;',
        timing='BEFORE',
        event='UPDATE',
        for_each='ROW'
    )
    logging.info("Trigger tg_set_updated_at_product created.")

    # Close database connection
    db.close()