import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def deployProd_step(accuracy_file,test_dir, compute_target,runconfig):   

    prod_deploy = PipelineParameter(name='prod_deploy', default_value=0)
    endpoint_traffic_pct = PipelineParameter(name='endpoint_traffic_pct', default_value=10)
    set_endpoint_as_default = PipelineParameter(name='set_endpoint_as_default', default_value=0)

    prod_scoring_url = PipelineData(
        name='prod_scoring_url', 
        pipeline_output_name='prod_scoring_url',
        datastore=accuracy_file.datastore,
        output_mode='mount',
        is_directory=False)

    outputs = [prod_scoring_url]
    outputs_map = { 'prod_scoring_url': prod_scoring_url }

    step = PythonScriptStep(
        script_name='deployProd.py',
        arguments=[           
            '--prod_scoring_url', prod_scoring_url,
            '--prod_deploy', prod_deploy,
            '--endpoint_traffic_pct', endpoint_traffic_pct,
            '--set_endpoint_as_default', set_endpoint_as_default
        ],
        inputs=[accuracy_file, test_dir],
        outputs=outputs,
        compute_target=compute_target,
        runconfig=runconfig,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )

    return step, outputs_map