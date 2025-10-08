import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_file_exist(self) -> bool:
        try:
            validation_status = True  # Start with True
            
            # Check if the directory exists
            if not os.path.exists(self.config.DATA_DIR):
                validation_status = False
                with open(self.config.STATUS_FILE, "w") as f:
                    f.write(f"validation_status: {validation_status}\n")
                    f.write(f"Error: Directory {self.config.DATA_DIR} does not exist")
                return validation_status
            
            all_files = os.listdir(self.config.DATA_DIR)
            
            # Check if all required files exist
            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    validation_status = False
                    break
            
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"validation_status: {validation_status}\n")
                if validation_status:
                    f.write(f"All required files are present: {self.config.ALL_REQUIRED_FILES}")
                else:
                    f.write(f"Missing files. Found: {all_files}\nRequired: {self.config.ALL_REQUIRED_FILES}")
            
            return validation_status
            
        except Exception as e:
            raise e