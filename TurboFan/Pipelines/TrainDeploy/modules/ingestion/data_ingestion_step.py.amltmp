import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def data_ingestion_step(datastore_reference, compute_target,runconfig):
    #run_config = RunConfiguration()
    #run_config.environment.docker.enabled = True

    raw_data_dir = PipelineData(
        name='raw_data_dir', 
        pipeline_output_name='raw_data_dir',
        datastore=datastore_reference.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [raw_data_dir]
    outputs_map = { 'raw_data_dir': raw_data_dir }

    step = PythonScriptStep(
        script_name='data_ingestion.py',
        arguments=['--output_dir', raw_data_dir, ],
        inputs=[datastore_reference],
        outputs=outputs,
        compute_target=compute_target,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        runconfig=runconfig,
        allow_reuse=True
    )

    return step, outputs_map