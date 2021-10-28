import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def deployTest_step(model_dir, accuracy_file, test_dir, compute_target,runconfig):   

    prod_deploy = PipelineParameter(name='prod_deploy', default_value=0)

    scoring_url = PipelineData(
        name='scoring_url', 
        pipeline_output_name='scoring_url',
        datastore=accuracy_file.datastore,
        output_mode='mount',
        is_directory=False)

    outputs = [scoring_url]
    outputs_map = { 'scoring_url': scoring_url }

    step = PythonScriptStep(
        script_name='deployTest.py',
        arguments=[
            '--test_dir', test_dir, 
            '--model_dir', model_dir, 
            '--accuracy_file', accuracy_file,             
            '--scoring_url', scoring_url,
            '--prod_deploy', prod_deploy
        ],
        inputs=[model_dir, accuracy_file, test_dir],
        outputs=outputs,
        compute_target=compute_target,
        runconfig=runconfig,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )

    return step, outputs_map