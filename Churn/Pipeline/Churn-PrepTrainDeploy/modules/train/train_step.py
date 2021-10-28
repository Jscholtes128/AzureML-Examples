import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
#from azureml.pipeline.steps import EstimatorStep
#from azureml.train.sklearn import SKLearn
#from sklearn.ensemble import GradientBoostingRegressor


def train_step(train_dir, compute_target,runconfig):

    n_estimators = PipelineParameter(name='n_estimators', default_value=50)
  
    model_dir = PipelineData(
        name='model_dir', 
        pipeline_output_name='model_dir',
        datastore=train_dir.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [model_dir]
    outputs_map = { 'model_dir': model_dir }


    step = PythonScriptStep(
        name="train_data",
        script_name='train.py',
        arguments=[
        '--train_dir', train_dir, 
        '--output_dir', model_dir, 
        '--n_estimators', n_estimators
        ],
        inputs=[train_dir],
        compute_target=compute_target,       
        outputs=outputs,
        runconfig=runconfig,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )


    return step, outputs_map