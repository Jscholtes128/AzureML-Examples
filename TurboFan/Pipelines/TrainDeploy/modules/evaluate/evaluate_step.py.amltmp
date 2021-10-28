import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
#from azureml.pipeline.steps import EstimatorStep
#from azureml.train.sklearn import SKLearn

def evaluate_step(model_dir, test_dir, compute_target,runconfig):   

    accuracy_file = PipelineData(
        name='accuracy_file', 
        pipeline_output_name='accuracy_file',
        datastore=test_dir.datastore,
        output_mode='mount',
        is_directory=False)

    outputs = [accuracy_file]
    outputs_map = { 'accuracy_file': accuracy_file }
    
   # estimator = SKLearn(
   #     source_directory=os.path.dirname(os.path.abspath(__file__)),
    #    entry_script='evaluate.py',
   #     compute_target=compute_target)

   # step = EstimatorStep(
   #     estimator=estimator,
   #     estimator_entry_script_arguments=[
   #         '--test_dir', test_dir, 
   #         '--model_dir', model_dir, 
   #         '--accuracy_file', accuracy_file
   #     ],
   #     inputs=[model_dir, test_dir],
   #     outputs=outputs,
   #     compute_target=compute_target,
   #     allow_reuse=True)


    step = PythonScriptStep(
    name="eval",
    script_name='evaluate.py',
    arguments=[
    '--test_dir', test_dir, 
        '--model_dir', model_dir, 
        '--accuracy_file', accuracy_file
    ],
    inputs=[model_dir, test_dir],
    outputs=outputs,
    compute_target=compute_target,
    runconfig=runconfig,
    source_directory=os.path.dirname(os.path.abspath(__file__)),
    allow_reuse=False
    )

    return step, outputs_map