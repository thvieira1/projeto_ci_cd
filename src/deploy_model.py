from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig

ws = Workspace.from_config()

model = Model(ws, name='my_model')

env = Environment.from_conda_specification(name="myenv", file_path="environment.yml")

inference_config = InferenceConfig(entry_script="scripts/score.py", environment=env)

aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name='my-service',
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config)

service.wait_for_deployment(show_output=True)
print(service.scoring_uri)