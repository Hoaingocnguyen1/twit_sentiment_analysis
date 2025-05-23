from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import logging


class MLSetupOperator(BaseOperator):    
    @apply_defaults
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import setup_environment
        return setup_environment(config=self.config, **context)

class MLDataLoaderOperator(BaseOperator):
    """Custom operator for loading and preprocessing data"""
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import load_and_preprocess_data
        return load_and_preprocess_data(**context)


class ChampionLoaderOperator(BaseOperator):
    """Custom operator for loading ML models"""
    
    @apply_defaults
    def __init__(self, model_name, base_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.base_model_path = base_model_path
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import load_champion_model
        return load_champion_model(
            model_name=self.model_name,
            **context
        )

class ChampionValidationOperator(BaseOperator):
    """Custom operator for model validation"""
    
    @apply_defaults
    def __init__(self, min_accuracy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_accuracy = min_accuracy
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import evaluate_champion_model
        return evaluate_champion_model(model_name=self.model_name, min_accuracy=self.min_accuracy, **context)

class CheckTrainingOperator(BaseOperator):
    """Custom operator for model validation"""
    
    @apply_defaults
    def __init__(self, min_accuracy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_accuracy = min_accuracy
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import check_training_decision
        return check_training_decision(**context)


class MLTrainingOperator(BaseOperator):
    """Custom operator for model training"""
    
    @apply_defaults
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import train_challenger_model
        return train_challenger_model(model_name=self.model_name, **context)


class MLModelRegistrationOperator(BaseOperator):
    @apply_defaults
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import register_challenger_model
        return register_challenger_model(model_name=self.model_name, **context)


class MLValidationOperation(BaseOperator):
    @apply_defaults
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import validate_challenger_model
        return validate_challenger_model(model_name=self.model_name, **context)

class MLPromotionOperator(BaseOperator):
    """Custom operator for model promotion (challenger/champion)"""
    
    @apply_defaults
    def __init__(self, model_name, promotion_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
    
    def execute(self, context):
        from plugins.scripts.ml_scripts import promote_challenger_to_champion
        return promote_challenger_to_champion(
            model_name=self.model_name,
            **context
        )