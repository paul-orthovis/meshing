import boto3

boto_session = boto3.session.Session(region_name='eu-west-1', profile_name='AdministratorAccess-643058308155')
sm_runtime = boto_session.client("sagemaker-runtime")

endpoint_name = "orthovis-dev-meshing"
input_location = "s3://orthovis-dev-case-uploads/6887ea037321894a9fd1d359/cropped-dicom.zip"

response = sm_runtime.invoke_endpoint_async(
    EndpointName=endpoint_name, 
    InputLocation=input_location,
    CustomAttributes='{"caseId": "6885783ff97431bc43a5eecf", "userId": "685d537073a9c240b3e8daec"}',
)
output_location = response['OutputLocation']
print(f"OutputLocation: {output_location}")

