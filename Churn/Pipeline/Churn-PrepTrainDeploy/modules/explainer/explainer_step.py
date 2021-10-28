import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
#from azureml.pipeline.steps import EstimatorStep
#from azureml.train.sklearn import SKLearn

def explainer_step(model_dir, test_dir,train_dir, compute_target,runconfig):   


    step = PythonScriptStep(
    name="explainer",
    script_name='explainer.py',
    arguments=[
    '--test_dir', test_dir, 
        '--model_dir', model_dir, 
        '--train_dir', train_dir
    ],
    inputs=[model_dir, test_dir,train_dir],
    compute_target=compute_target,
    runconfig=runconfig,
    source_directory=os.path.dirname(os.path.abspath(__file__)),
    allow_reuse=False
    )

    return step