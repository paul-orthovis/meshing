import os
import shutil
import tarfile
import tempfile
import argparse

import boto3
import sagemaker
from sagemaker.s3 import S3Uploader


role = 'arn:aws:iam::643058308155:role/SageMakerExecutionRole'
variant_name = "variant1"


def get_model_name(stack):
    # FIXME: add a version number to the model name
    return f'orthovis-{stack}-meshing'

def get_endpoint_config_name(stack):
    return f'orthovis-{stack}-meshing'

def get_endpoint_name(stack):
    return f'orthovis-{stack}-meshing'

def get_resource_id_for_autoscaling(stack):
    return f'endpoint/{get_endpoint_name(stack)}/variant/{variant_name}'

def get_sns_topics(stack):
    return {
        'SuccessTopic': f'arn:aws:sns:eu-west-1:643058308155:orthovis-{stack}-meshing-completed', 
        'ErrorTopic': f'arn:aws:sns:eu-west-1:643058308155:orthovis-{stack}-meshing-failed',
    }

def create_sagemaker_model_package(model_folder_path):
    """
    Create a SageMaker model package with the required folder structure.
    
    This function creates a tar.gz file with the following structure:
    ```
    model_and_code.tar.gz
    ├── code/
    │   ├── requirements.txt
    │   ├── sagemaker_entrypoint.py
    │   └── meshing.py
    └── model/
        └── [nnUNet model files]
    ```
    
    Args:
        model_folder_path: Path to the nnUNet model folder
    
    Returns:
        Path to the created tar.gz file
    """
    # Create temporary directory for the model package
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the SageMaker required folder structure
        code_dir = os.path.join(temp_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)
        
        # Copy required files from current directory
        required_files = ['requirements.txt', 'inference.py', 'meshing.py']
        for src_path in required_files:
            dst_path = os.path.join(code_dir, src_path)
            shutil.copy2(src_path, dst_path)
        
        # Copy model folder to the model directory
        # FIXME: only copy the relevant stuff (so not progress.png and similar -- see the nnunet export script; restrict to one fold if relevant!)
        model_dir = os.path.join(temp_dir, 'model')
        shutil.copytree(model_folder_path, model_dir)
        
        # Create .tar.gz file
        filename = 'model_and_code.tar.gz'
        with tarfile.open(filename, 'w:gz') as tar:
            # Add the contents of temp_dir directly to the root of the tar
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                tar.add(item_path, arcname=item)
        
        return filename


def deploy(sm_session, sm_client, autoscaling_client, cw_client, boto_session, stack, nnunet_path):

    sm_bucket = sm_session.default_bucket()
    region = boto_session.region_name

    filename = create_sagemaker_model_package(nnunet_path)
    model_artifact = S3Uploader.upload(filename, f's3://{sm_bucket}/{stack}/meshing', sagemaker_session=sm_session)
    print(model_artifact)

    instance_type = 'ml.g4dn.2xlarge'

    image_uri = sagemaker.image_uris.retrieve(
        'pytorch',
        region,
        version='2.4',
        py_version='py311',
        instance_type=instance_type,
        accelerator_type=None,  # only used for elastic inference
        image_scope='inference'
    )
    print(image_uri)

    model_name = get_model_name(stack)
    endpoint_config_name = get_endpoint_config_name(stack)
    endpoint_name = get_endpoint_name(stack)

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_artifact,
            'Environment': {
                'TS_MAX_REQUEST_SIZE': str(1024 ** 3),  # default max request size is 6 Mb for torchserve, need to increase
                'TS_MAX_RESPONSE_SIZE': str(1024 ** 3),
                'TS_DEFAULT_RESPONSE_TIMEOUT': '600',
            }
        },
    )
    if create_model_response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('create_model failed:', create_model_response)

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": variant_name,
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://orthovis-{stack}-meshing-results",
                "NotificationConfig": get_sns_topics(stack)
            },
            "ClientConfig": {
                "MaxConcurrentInvocationsPerInstance": 2,
            }
        }
    )
    if create_endpoint_config_response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('create_endpoint_config failed:', create_endpoint_config_response)
    print(f"Created EndpointConfig: {create_endpoint_config_response['EndpointConfigArn']}")

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    if create_endpoint_response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('create_endpoint failed:', create_endpoint_response)
    print(f"Creating Endpoint: {create_endpoint_response['EndpointArn']}")
    waiter = sm_client.get_waiter('endpoint_in_service')
    print("Waiting for endpoint to create...")
    waiter.wait(EndpointName=endpoint_name)
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    if resp['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('describe_endpoint failed:', resp)
    print(f"Endpoint Status: {resp['EndpointStatus']}")

    resource_id_for_autoscaling = get_resource_id_for_autoscaling(stack)

    response = autoscaling_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id_for_autoscaling,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=0,
        MaxCapacity=5
    )
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('register_scalable_target failed:', response)
    print(f"Registered scalable target: {response['ScalableTargetARN']}")

    response = autoscaling_client.put_scaling_policy(
        PolicyName='Invocations-ScalingPolicy',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id_for_autoscaling,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 5.0,
            'CustomizedMetricSpecification': {
                'MetricName': 'ApproximateBacklogSizePerInstance',
                'Namespace': 'AWS/SageMaker',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name }
                ],
                'Statistic': 'Average',
            },
            'ScaleInCooldown': 120, # The cooldown period helps you prevent your Auto Scaling group from launching or terminating
                                    # additional instances before the effects of previous activities are visible.
                                    # You can configure the length of time based on your instance startup time or other application needs.
                                    # ScaleInCooldown - The amount of time, in seconds, after a scale in activity completes before another scale in activity can start.
            'ScaleOutCooldown': 120 # ScaleOutCooldown - The amount of time, in seconds, after a scale out activity completes before another scale out activity can start.
        }
    )
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('put_scaling_policy (Invocations-ScalingPolicy) failed:', response)
    print(f"Created backlog-sizescaling policy: {response['PolicyARN']}")


    response = autoscaling_client.put_scaling_policy(
        PolicyName="HasBacklogWithoutCapacity-ScalingPolicy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id_for_autoscaling,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity", # Specifies whether the ScalingAdjustment value in the StepAdjustment property is an absolute number or a percentage of the current capacity.
            "MetricAggregationType": "Average", # The aggregation type for the CloudWatch metrics.
            "Cooldown": 120, # The amount of time, in seconds, to wait for a previous scaling activity to take effect.
            "StepAdjustments": # A set of adjustments that enable you to scale based on the size of the alarm breach.
            [
                {
                  "MetricIntervalLowerBound": 0,
                  "ScalingAdjustment": 1
                }
              ]
        },
    )
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('put_scaling_policy (HasBacklogWithoutCapacity-ScalingPolicy) failed:', response)
    print(f"Created no-capacity scaling policy: {response['PolicyARN']}")
    step_scaling_policy_arn = response['PolicyARN']

    response = cw_client.put_metric_alarm(
        AlarmName=f"HasBacklogWithoutCapacity-{resource_id_for_autoscaling}",
        MetricName='HasBacklogWithoutCapacity',
        Namespace='AWS/SageMaker',
        Statistic='Average',
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold= 1,
        ComparisonOperator='GreaterThanOrEqualToThreshold',
        TreatMissingData='missing',
        Dimensions=[
            { 'Name':'EndpointName', 'Value':endpoint_name },
        ],
        Period=30,
        AlarmActions=[step_scaling_policy_arn]
    )
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('put_metric_alarm failed:', response)
    print(f"Created alarm for no-capacity scaling policy")


def undeploy(sm_client, autoscaling_client, cw_client, stack):

    endpoint_name = get_endpoint_name(stack)
    endpoint_config_name = get_endpoint_config_name(stack)
    model_name = get_model_name(stack)
    resource_id_for_autoscaling = get_resource_id_for_autoscaling(stack)
    
    # Wait till the endpoint is in service (not creating/updating, so we can delete it)
    print(f"Waiting for endpoint {endpoint_name} to be in service...")
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    
    # Deregister scalable target
    response = autoscaling_client.deregister_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id_for_autoscaling,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount'
    )
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('deregister_scalable_target failed:', response)
    else:
        print(f"Deregistered scalable target: {resource_id_for_autoscaling}")

    # Delete CloudWatch alarm
    response = cw_client.delete_alarms(AlarmNames=[f"HasBacklogWithoutCapacity-{resource_id_for_autoscaling}"])
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('delete_alarms failed:', response)
    else:
        print(f"Deleted CloudWatch alarm: HasBacklogWithoutCapacity-{resource_id_for_autoscaling}")

    # Delete endpoint
    response = sm_client.delete_endpoint(EndpointName=endpoint_name)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('delete_endpoint failed:', response)
    else:
        print(f"Deleted endpoint: {endpoint_name}")

    # Delete endpoint config
    response = sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('delete_endpoint_config failed:', response)
    else:
        print(f"Deleted endpoint config: {endpoint_config_name}")

    # Delete model
    response = sm_client.delete_model(ModelName=model_name)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('delete_model failed:', response)
    else:
        print(f"Deleted model: {model_name}")


def main():

    parser = argparse.ArgumentParser(description='Deploy or undeploy SageMaker endpoint')
    parser.add_argument('--undeploy', action='store_true', help='Undeploy the endpoint')
    parser.add_argument('--region', type=str, default='eu-west-1', help='AWS region name')
    parser.add_argument('--stack', type=str, required=True, choices=['test', 'dev', 'prod'], help='Stack environment')
    parser.add_argument('--nnunet-path', type=str, help='Path to the nnUNet model folder')
    parser.add_argument('--profile', type=str, help='AWS profile name')
    args = parser.parse_args()
    
    boto_session = boto3.session.Session(region_name=args.region, profile_name=args.profile)
    sm_session = sagemaker.session.Session(boto_session)
    sm_client = boto_session.client("sagemaker")
    autoscaling_client = boto_session.client('application-autoscaling')
    cw_client = boto_session.client('cloudwatch')
    
    if args.undeploy:
        undeploy(sm_client, autoscaling_client, cw_client, args.stack)
    else:
        assert args.nnunet_path is not None, "--nnunet-path is required for deployment"
        deploy(sm_session, sm_client, autoscaling_client, cw_client, boto_session, args.stack, args.nnunet_path)


if __name__ == "__main__":
    main()
    
    