from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer .components.model_evalution import ModelEvalution
from textSummarizer.logging import logger

class ModelEvalutionPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evalution_config = config.get_model_evalution_config()
        model_valution_config = ModelEvalution(config=model_evalution_config)
        model_evalution_config.train()

    