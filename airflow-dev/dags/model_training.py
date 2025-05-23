from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

# Import custom operators
from plugins.operators.ml_operator import (
    MLSetupOperator,
    MLDataLoaderOperator, 
    ChampionLoaderOperator,
    ChampionValidationOperator,
    CheckTrainingOperator,
    MLTrainingOperator,
    MLModelRegistrationOperator,
    MLValidationOperation,
    MLPromotionOperator
)

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 23, 5),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG configuration
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML Model Training and Validation Pipeline',
    schedule_interval='@Weekly',
    max_active_runs=1,
    tags=['ml', 'training', 'champion-challenger']
)

# Configuration for ML pipeline
ml_config = {
    'model_type': 'classification',
    'feature_columns': ['feature1', 'feature2', 'feature3'],
    'target_column': 'target',
    'test_size': 0.2,
    'random_state': 42
}

# 1. Setup environment
setup_task = MLSetupOperator(
    task_id='setup_ml_environment',
    config=ml_config,
    dag=dag
)

# 2. Load and preprocess data
data_loader_task = MLDataLoaderOperator(
    task_id='load_preprocess_data',
    dag=dag
)

# 3. Load current champion model
champion_loader_task = ChampionLoaderOperator(
    task_id='load_champion_model',
    model_name='my_ml_model',
    base_model_path='/models/champion/',
    dag=dag
)

# 4. Validate champion model performance
champion_validation_task = ChampionValidationOperator(
    task_id='validate_champion_model',
    min_accuracy=0.85,
    dag=dag
)

# 5. Check if training is needed
def check_training_decision(**context):
    """Decision function to determine if new training is needed"""
    # Logic to check if training is required
    # This could be based on performance metrics, data drift, etc.
    champion_metrics = context['task_instance'].xcom_pull(task_ids='validate_champion_model')
    
    if champion_metrics and champion_metrics.get('accuracy', 0) < 0.85:
        return 'train_challenger_model'
    else:
        return 'skip_training'

training_decision_task = BranchPythonOperator(
    task_id='check_training_decision',
    python_callable=check_training_decision,
    dag=dag
)

# 6. Train challenger model (conditional)
train_challenger_task = MLTrainingOperator(
    task_id='train_challenger_model',
    config=ml_config,
    dag=dag
)

# 7. Register challenger model
register_challenger_task = MLModelRegistrationOperator(
    task_id='register_challenger_model',
    model_name='my_ml_model_challenger',
    dag=dag
)

# 8. Validate challenger model
validate_challenger_task = MLValidationOperation(
    task_id='validate_challenger_model',
    model_name='my_ml_model_challenger',
    dag=dag
)

# 9. Promotion decision
def promotion_decision(**context):
    """Decision function to determine if challenger should be promoted"""
    challenger_metrics = context['task_instance'].xcom_pull(task_ids='validate_challenger_model')
    champion_metrics = context['task_instance'].xcom_pull(task_ids='validate_champion_model')
    
    challenger_accuracy = challenger_metrics.get('accuracy', 0) if challenger_metrics else 0
    champion_accuracy = champion_metrics.get('accuracy', 0) if champion_metrics else 0
    
    # Promote if challenger performs better than champion
    if challenger_accuracy > champion_accuracy:
        return 'promote_challenger_to_champion'
    else:
        return 'keep_current_champion'

promotion_decision_task = BranchPythonOperator(
    task_id='promotion_decision',
    python_callable=promotion_decision,
    dag=dag
)

# 10. Promote challenger to champion
promote_challenger_task = MLPromotionOperator(
    task_id='promote_challenger_to_champion',
    model_name='my_ml_model_challenger',
    promotion_type='champion',
    dag=dag
)

# 11. Dummy tasks for flow control
skip_training_task = DummyOperator(
    task_id='skip_training',
    dag=dag
)

keep_champion_task = DummyOperator(
    task_id='keep_current_champion',
    dag=dag
)

pipeline_complete_task = DummyOperator(
    task_id='pipeline_complete',
    trigger_rule='none_failed_or_skipped',  # Continue if any upstream task succeeds
    dag=dag
)

# Define task dependencies
setup_task >> data_loader_task >> champion_loader_task >> champion_validation_task

champion_validation_task >> training_decision_task

# Training branch
training_decision_task >> train_challenger_task >> register_challenger_task
register_challenger_task >> validate_challenger_task >> promotion_decision_task

# Promotion branches
promotion_decision_task >> [promote_challenger_task, keep_champion_task]

# Skip training branch
training_decision_task >> skip_training_task

# All paths converge to completion
[promote_challenger_task, keep_champion_task, skip_training_task] >> pipeline_complete_task