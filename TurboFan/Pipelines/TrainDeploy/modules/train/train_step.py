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

    max_depth = PipelineParameter(name='max_depth', default_value=5)
    n_estimators = PipelineParameter(name='n_estimators', default_value=500)
  
    model_dir = PipelineData(
        name='model_dir', 
        pipeline_output_name='model_dir',
        datastore=train_dir.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [model_dir]
    outputs_map = { 'model_dir': model_dir }

    #estimator = SKLearn(
    #    source_directory=os.path.dirname(os.path.abspath(__file__)),
    #    entry_script='train.py',
    #    compute_target=compute_target
   #     )



    step = PythonScriptStep(
        name="train_data",
        script_name='train.py',
        arguments=[
        '--train_dir', train_dir, 
        '--output_dir', model_dir, 
        '--max_depth', max_depth, 
        '--n_estimators', n_estimators
        ],
        inputs=[train_dir],
        compute_target=compute_target,       
        outputs=outputs,
        runconfig=runconfig,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )
   # step = EstimatorStep(
    #    estimator=estimator,
    #    estimator_entry_script_arguments=[
     #       '--train_dir', train_dir, 
    #        '--output_dir', model_dir, 
    #        '--max_depth', max_depth, 
    #        '--n_estimators', n_estimators
    #    ],
    #    inputs=[train_dir],
   #     compute_target=compute_target,       
    #    outputs=outputs,
    #    allow_reuse=False)

    return step, outputs_map