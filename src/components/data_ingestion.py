import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')
    source_data_path: str = os.path.join('notebook', 'data', 'gemstone.csv')

class DataIngestion:
    def __init__(self, ingestion_config: Optional[DataIngestionConfig] = None):
        self.ingestion_config = ingestion_config or DataIngestionConfig()

    def _resolve_source_path(self) -> Path:
        source_path = Path(self.ingestion_config.source_data_path)
        if source_path.exists():
            return source_path

        fallback_path = Path('data') / 'gemstone.csv'
        if fallback_path.exists():
            return fallback_path

        raise FileNotFoundError(
            f"No source dataset found at {source_path} or {fallback_path}"
        )

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logging.info('Data ingestion method Started')
        try:
            source_path = self._resolve_source_path()
            df = pd.read_csv(source_path)
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)

    # Backward compatibility for legacy calls.
    def initate_data_ingestion(self) -> Tuple[str, str]:
        return self.initiate_data_ingestion()
    
# Run Data ingestion
if __name__ == '__main__':
    from src.pipeline.train_pipeline import run_training_pipeline

    run_training_pipeline()