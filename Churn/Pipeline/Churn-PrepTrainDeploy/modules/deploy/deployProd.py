import argparse
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksEndpoint
from azureml.core.compute import AksCompute
from azureml.core.compute import ComputeTarget
from azureml.exceptions import WebserviceException


# Define arguments
parser = argparse.ArgumentParser(description='Deploy arg parser')
parser.add_argument('--prod_scoring_url', type=str, help='File storing the scoring url')
parser.add_argument('--prod_deploy', type=int, help='Is Prod Deploy')
parser.add_argument('--endpoint_traffic_pct', type=int, help='Is Prod Deploy')
parser.add_argument('--set_endpoint_as_default', type=int, help='Is Prod Deploy')


args = parser.parse_args()

# Get arguments from parser

scoring_url = args.prod_scoring_url
prod_deploy = args.prod_deploy
endpoint_traffic_pct = args.endpoint_traffic_pct
set_endpoint_as_default = args.set_endpoint_as_default

run = Run.get_context()
ws = run.experiment.workspace

if prod_deploy == 1:
    # Define model and service names
    endpoint_name = 'turbofanprod'
    model_name = 'turbofan-pipeline-rul'

    # Get run context
    
    model = Model(ws,model_name)

    compute = ComputeTarget(ws, 'aksclst')

    # Create inference config
    inference_config = InferenceConfig(
        source_directory = '.',
        runtime = 'python', 
        entry_script = 'score.py',
        conda_file = 'turbofan.yml')

    version_name= "turbofan" + str(model.version)
    print("version name: " + str(version_name))
    # create the deployment config and define the scoring traffic percentile for the first deployment
    endpoint_deployment_config = AksEndpoint.deploy_configuration(cpu_cores = 0.1, memory_gb = 0.2,
                                enable_app_insights = True,
                                collect_model_data=True,  
                                auth_enabled=True,  
                                description = "Nasa Turbofan",
                                version_name = str(version_name),
                                traffic_percentile = int(endpoint_traffic_pct))
                                #is_default=set_endpoint_as_default)


    
    endpoint_exisits = False

    try:
        endpoint = AksEndpoint(ws, name=endpoint_name)
        if endpoint:
            endpoint_exisits = True
    except WebserviceException as e:
        print("Endpoint does not exist")
    

    if endpoint_exisits:
        if version_name in endpoint.versions:
            endpoint.update_version(version_name = version_name,
                        traffic_percentile=int(endpoint_traffic_pct),
                       is_default=bool(set_endpoint_as_default),
                       is_control_version_type=True)
        else:
            endpoint.create_version(version_name = version_name,
                       inference_config=inference_config,
                       models=[model],
                       tags = {'modelVersion':version_name},
                       description = "turbofan " + version_name,
                       traffic_percentile = int(endpoint_traffic_pct))
        
    else:
        # deploy the model and endpoint
        endpoint = Model.deploy(ws, endpoint_name, [model], inference_config, endpoint_deployment_config,compute)
    # Wait for he process to complete
    endpoint.wait_for_deployment(True)

    # Output scoring url
    print(endpoint.scoring_uri)
    with open(scoring_url, 'w+') as f:
        f.write(endpoint.scoring_uri)
else:
    with open(scoring_url, 'w+') as f:
        f.write('')
