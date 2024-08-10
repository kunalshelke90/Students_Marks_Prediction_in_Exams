import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("Starting data transformation")
            train_array, test_array,_= self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            
            # Step 3: Model Training
            logging.info("Starting model training")
            model_score = self.model_trainer.initiate_model_trainer(train_array, test_array)
            
            logging.info(f"Training pipeline completed successfully with model score: {model_score}")
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
