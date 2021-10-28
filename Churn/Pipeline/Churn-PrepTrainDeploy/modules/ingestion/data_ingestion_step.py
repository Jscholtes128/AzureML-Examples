import os
from azureml.pipeline.steps import PythonScriptStep,DatabricksStep
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def data_ingestion_step(datastore_reference, compute_target):

    raw_data_dir = PipelineData(
        name='raw_data_dir', 
        pipeline_output_name='raw_data_dir',
        datastore=datastore_reference.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [raw_data_dir]
    outputs_map = { 'raw_data_dir': raw_data_dir }

    
    step = DatabricksStep(
    name="adb-churn-ingest",
    notebook_path='/Users/joscholt@microsoft.com/Examples/CustomerChurn/01-Churn-DataLoad',
    run_name='churnpipeline',
    outputs=outputs,
    inputs=[datastore_reference],
    compute_target=compute_target,
    existing_cluster_id="1025-153730-47woh1e0",
    allow_reuse=False,
    permit_cluster_restart=True)

    return step, raw_data_dir