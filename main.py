from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from textSummarizer.pipeline.stage_05_model_evalution import ModelEvalutionPipeline
from textSummarizer.logging import logger



STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evalution stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    model_evalution = ModelEvalutionPipeline()
    model_evalution.main()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e