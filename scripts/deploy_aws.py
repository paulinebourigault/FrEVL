#!/usr/bin/env python3
"""
Deploy FrEVL to AWS
Automated deployment to Amazon Web Services (ECS, EKS, or SageMaker)
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError


class AWSDeployer:
    """Deploy FrEVL to various AWS services"""
    
    def __init__(
        self,
        region: str = "us-west-2",
        profile: Optional[str] = None
    ):
        self.region = region
        
        # Initialize AWS clients
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        
        session = boto3.Session(**session_kwargs)
        
        self.ecr_client = session.client('ecr', region_name=region)
        self.ecs_client = session.client('ecs', region_name=region)
        self.eks_client = session.client('eks', region_name=region)
        self.sagemaker_client = session.client('sagemaker', region_name=region)
        self.s3_client = session.client('s3', region_name=region)
        self.iam_client = session.client('iam', region_name=region)
        
        # Get account ID
        self.account_id = session.client('sts').get_caller_identity()['Account']
    
    # ============================================================================
    # ECR (Elastic Container Registry)
    # ============================================================================
    
    def create_ecr_repository(self, repository_name: str) -> str:
        """Create ECR repository for Docker images"""
        
        print(f"Creating ECR repository: {repository_name}")
        
        try:
            response = self.ecr_client.create_repository(
                repositoryName=repository_name,
                imageScanningConfiguration={
                    'scanOnPush': True
                },
                encryptionConfiguration={
                    'encryptionType': 'AES256'
                }
            )
            repository_uri = response['repository']['repositoryUri']
            print(f" Created repository: {repository_uri}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'RepositoryAlreadyExistsException':
                response = self.ecr_client.describe_repositories(
                    repositoryNames=[repository_name]
                )
                repository_uri = response['repositories'][0]['repositoryUri']
                print(f" Repository exists: {repository_uri}")
            else:
                raise
        
        return repository_uri
    
    def push_to_ecr(self, repository_name: str, local_image: str) -> str:
        """Push Docker image to ECR"""
        
        repository_uri = self.create_ecr_repository(repository_name)
        
        # Get ECR login token
        print("Getting ECR login token...")
        response = self.ecr_client.get_authorization_token()
        auth_data = response['authorizationData'][0]
        
        # Login to ECR
        import base64
        token = base64.b64decode(auth_data['authorizationToken']).decode('utf-8')
        username, password = token.split(':')
        registry = auth_data['proxyEndpoint'].replace('https://', '')
        
        # Docker commands
        print("Logging into ECR...")
        os.system(f"echo {password} | docker login --username {username} --password-stdin {registry}")
        
        # Tag and push image
        print(f"Tagging image...")
        os.system(f"docker tag {local_image} {repository_uri}:latest")
        
        print(f"Pushing image to ECR...")
        os.system(f"docker push {repository_uri}:latest")
        
        print(f" Image pushed to {repository_uri}")
        return repository_uri
    
    # ============================================================================
    # ECS (Elastic Container Service)
    # ============================================================================
    
    def deploy_to_ecs(
        self,
        cluster_name: str,
        service_name: str,
        task_family: str,
        image_uri: str,
        cpu: str = "4096",
        memory: str = "8192",
        gpu: int = 0
    ):
        """Deploy to ECS using Fargate or EC2"""
        
        print(f"Deploying to ECS cluster: {cluster_name}")
        
        # Create cluster if not exists
        try:
            self.ecs_client.create_cluster(clusterName=cluster_name)
            print(f"✓ Created cluster: {cluster_name}")
        except ClientError:
            print(f"✓ Using existing cluster: {cluster_name}")
        
        # Create task definition
        task_definition = {
            'family': task_family,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'] if gpu == 0 else ['EC2'],
            'cpu': cpu,
            'memory': memory,
            'containerDefinitions': [
                {
                    'name': 'frevl',
                    'image': image_uri,
                    'essential': True,
                    'portMappings': [
                        {
                            'containerPort': 8000,
                            'protocol': 'tcp'
                        }
                    ],
                    'environment': [
                        {'name': 'FREVL_MODEL', 'value': 'frevl-base'},
                        {'name': 'DEVICE', 'value': 'cuda' if gpu > 0 else 'cpu'},
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': f'/ecs/{task_family}',
                            'awslogs-region': self.region,
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }
            ]
        }
        
        # Add GPU if requested
        if gpu > 0:
            task_definition['containerDefinitions'][0]['resourceRequirements'] = [
                {
                    'type': 'GPU',
                    'value': str(gpu)
                }
            ]
        
        # Register task definition
        print("Registering task definition...")
        response = self.ecs_client.register_task_definition(**task_definition)
        task_definition_arn = response['taskDefinition']['taskDefinitionArn']
        print(f"✓ Registered task: {task_definition_arn}")
        
        # Create or update service
        print(f"Creating ECS service: {service_name}")
        
        try:
            response = self.ecs_client.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_family,
                desiredCount=2,
                launchType='FARGATE' if gpu == 0 else 'EC2',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': self._get_subnet_ids(),
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
            print(f"✓ Created service: {service_name}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidParameterException':
                # Update existing service
                response = self.ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    taskDefinition=task_family,
                    desiredCount=2
                )
                print(f" Updated service: {service_name}")
            else:
                raise
        
        print(f" Deployment to ECS complete!")
        return response['service']['serviceArn']
    
    # ============================================================================
    # SageMaker
    # ============================================================================
    
    def deploy_to_sagemaker(
        self,
        model_name: str,
        model_path: str,
        instance_type: str = "ml.g4dn.xlarge",
        endpoint_name: Optional[str] = None
    ):
        """Deploy model to SageMaker endpoint"""
        
        print(f"Deploying to SageMaker: {model_name}")
        
        # Upload model to S3
        bucket_name = f"frevl-models-{self.account_id}"
        model_key = f"models/{model_name}/model.tar.gz"
        
        print(f"Creating S3 bucket: {bucket_name}")
        try:
            self.s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )
        except ClientError:
            pass  # Bucket already exists
        
        # Package and upload model
        print("Packaging model...")
        os.system(f"cd {model_path} && tar -czf model.tar.gz *")
        
        print(f"Uploading model to S3...")
        self.s3_client.upload_file(
            f"{model_path}/model.tar.gz",
            bucket_name,
            model_key
        )
        
        model_data_url = f"s3://{bucket_name}/{model_key}"
        print(f" Model uploaded: {model_data_url}")
        
        # Get or create execution role
        role_arn = self._get_sagemaker_role()
        
        # Create model
        print("Creating SageMaker model...")
        
        primary_container = {
            'Image': f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/frevl-sagemaker:latest",
            'ModelDataUrl': model_data_url,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
            }
        }
        
        try:
            response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer=primary_container,
                ExecutionRoleArn=role_arn
            )
            print(f"✓ Created model: {model_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                print(f"✓ Model exists: {model_name}")
            else:
                raise
        
        # Create endpoint configuration
        endpoint_config_name = f"{model_name}-config"
        
        print("Creating endpoint configuration...")
        try:
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'default',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1
                    }
                ]
            )
            print(f"✓ Created endpoint config: {endpoint_config_name}")
        except ClientError:
            print(f"✓ Endpoint config exists: {endpoint_config_name}")
        
        # Create or update endpoint
        if endpoint_name is None:
            endpoint_name = f"{model_name}-endpoint"
        
        print(f"Creating endpoint: {endpoint_name}")
        try:
            response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print(f" Creating endpoint... (this may take 5-10 minutes)")
            
            # Wait for endpoint to be in service
            self._wait_for_endpoint(endpoint_name)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                # Update existing endpoint
                print(f"Updating existing endpoint: {endpoint_name}")
                response = self.sagemaker_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
                self._wait_for_endpoint(endpoint_name)
            else:
                raise
        
        # Get endpoint details
        response = self.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        
        print(f" Endpoint active: {response['EndpointArn']}")
        print(f" Status: {response['EndpointStatus']}")
        print(f" URL: https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{endpoint_name}/invocations")
        
        return response['EndpointArn']
    
    # ============================================================================
    # EKS (Elastic Kubernetes Service)
    # ============================================================================
    
    def deploy_to_eks(
        self,
        cluster_name: str,
        namespace: str = "frevl",
        deployment_file: str = "deploy/k8s/deployment.yaml"
    ):
        """Deploy to EKS cluster"""
        
        print(f"Deploying to EKS cluster: {cluster_name}")
        
        # Get cluster info
        try:
            response = self.eks_client.describe_cluster(name=cluster_name)
            cluster_info = response['cluster']
            print(f"✓ Found cluster: {cluster_name}")
            print(f"  Status: {cluster_info['status']}")
            print(f"  Endpoint: {cluster_info['endpoint']}")
            
        except ClientError:
            print(f" Cluster not found: {cluster_name}")
            print("Please create the cluster first using:")
            print(f"  eksctl create cluster --name {cluster_name} --region {self.region}")
            return
        
        # Update kubeconfig
        print("Updating kubeconfig...")
        os.system(f"aws eks update-kubeconfig --name {cluster_name} --region {self.region}")
        
        # Create namespace
        print(f"Creating namespace: {namespace}")
        os.system(f"kubectl create namespace {namespace} --dry-run=client -o yaml | kubectl apply -f -")
        
        # Deploy application
        print("Deploying application...")
        os.system(f"kubectl apply -f {deployment_file} -n {namespace}")
        
        # Wait for deployment
        print("Waiting for deployment to be ready...")
        os.system(f"kubectl wait --for=condition=available --timeout=300s deployment/frevl-api -n {namespace}")
        
        # Get service endpoint
        print("Getting service endpoint...")
        os.system(f"kubectl get service frevl-api -n {namespace}")
        
        print(f" Deployment to EKS complete!")
    
    # ============================================================================
    # Lambda (Serverless)
    # ============================================================================
    
    def deploy_to_lambda(
        self,
        function_name: str,
        model_path: str,
        memory_size: int = 3008,
        timeout: int = 60
    ):
        """Deploy lightweight model to Lambda"""
        
        print(f"Deploying to Lambda: {function_name}")
        
        lambda_client = boto3.client('lambda', region_name=self.region)
        
        # Package function
        print("Packaging Lambda function...")
        package_path = Path("lambda_package.zip")
        
        os.system(f"""
            mkdir -p lambda_package
            cp -r {model_path}/* lambda_package/
            cp deploy/lambda/handler.py lambda_package/
            cd lambda_package && zip -r ../{package_path} .
            cd .. && rm -rf lambda_package
        """)
        
        # Upload to S3 (Lambda has 50MB limit for direct upload)
        bucket_name = f"frevl-lambda-{self.account_id}"
        key = f"functions/{function_name}.zip"
        
        try:
            self.s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )
        except ClientError:
            pass
        
        print(f"Uploading to S3...")
        self.s3_client.upload_file(str(package_path), bucket_name, key)
        
        # Get or create execution role
        role_arn = self._get_lambda_role()
        
        # Create or update function
        print(f"Creating Lambda function...")
        
        try:
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='handler.lambda_handler',
                Code={
                    'S3Bucket': bucket_name,
                    'S3Key': key
                },
                Description='FrEVL inference endpoint',
                Timeout=timeout,
                MemorySize=memory_size,
                Environment={
                    'Variables': {
                        'MODEL_PATH': '/tmp/model',
                        'DEVICE': 'cpu'
                    }
                }
            )
            print(f" Created function: {function_name}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException':
                # Update existing function
                print(f"Updating existing function...")
                response = lambda_client.update_function_code(
                    FunctionName=function_name,
                    S3Bucket=bucket_name,
                    S3Key=key
                )
                
                lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Timeout=timeout,
                    MemorySize=memory_size
                )
                print(f" Updated function: {function_name}")
            else:
                raise
        
        # Create API Gateway
        print("Creating API Gateway...")
        self._create_api_gateway(function_name)
        
        print(f" Lambda deployment complete!")
        return response['FunctionArn']
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    def _get_subnet_ids(self) -> list:
        """Get default VPC subnet IDs"""
        ec2_client = boto3.client('ec2', region_name=self.region)
        
        response = ec2_client.describe_subnets(
            Filters=[
                {
                    'Name': 'default-for-az',
                    'Values': ['true']
                }
            ]
        )
        
        return [subnet['SubnetId'] for subnet in response['Subnets']][:2]
    
    def _get_sagemaker_role(self) -> str:
        """Get or create SageMaker execution role"""
        role_name = 'FrEVLSageMakerRole'
        
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            return response['Role']['Arn']
            
        except ClientError:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "sagemaker.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for FrEVL SageMaker'
            )
            
            # Attach policy
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            return response['Role']['Arn']
    
    def _get_lambda_role(self) -> str:
        """Get or create Lambda execution role"""
        role_name = 'FrEVLLambdaRole'
        
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            return response['Role']['Arn']
            
        except ClientError:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for FrEVL Lambda'
            )
            
            # Attach policies
            policies = [
                'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                'arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
            ]
            
            for policy in policies:
                self.iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy
                )
            
            return response['Role']['Arn']
    
    def _wait_for_endpoint(self, endpoint_name: str, max_wait: int = 600):
        """Wait for SageMaker endpoint to be in service"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            status = response['EndpointStatus']
            
            if status == 'InService':
                print(f" Endpoint is in service")
                return
            elif status in ['Failed', 'RollingBack']:
                raise Exception(f"Endpoint creation failed: {status}")
            
            print(f"  Status: {status}... waiting")
            time.sleep(30)
        
        raise TimeoutError(f"Endpoint creation timed out after {max_wait} seconds")
    
    def _create_api_gateway(self, function_name: str):
        """Create API Gateway for Lambda function"""
        api_client = boto3.client('apigateway', region_name=self.region)
        
        # Create REST API
        response = api_client.create_rest_api(
            name=f"{function_name}-api",
            description='FrEVL API Gateway',
            endpointConfiguration={
                'types': ['REGIONAL']
            }
        )
        
        api_id = response['id']
        
        # Get root resource
        resources = api_client.get_resources(restApiId=api_id)
        root_id = resources['items'][0]['id']
        
        # Create resource
        resource = api_client.create_resource(
            restApiId=api_id,
            parentId=root_id,
            pathPart='predict'
        )
        
        # Create method
        api_client.put_method(
            restApiId=api_id,
            resourceId=resource['id'],
            httpMethod='POST',
            authorizationType='NONE'
        )
        
        # Create integration
        lambda_client = boto3.client('lambda', region_name=self.region)
        function_arn = lambda_client.get_function(
            FunctionName=function_name
        )['Configuration']['FunctionArn']
        
        api_client.put_integration(
            restApiId=api_id,
            resourceId=resource['id'],
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=f"arn:aws:apigateway:{self.region}:lambda:path/2015-03-31/functions/{function_arn}/invocations"
        )
        
        # Deploy API
        api_client.create_deployment(
            restApiId=api_id,
            stageName='prod'
        )
        
        # Grant permission to API Gateway
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f"apigateway-{api_id}",
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=f"arn:aws:execute-api:{self.region}:{self.account_id}:{api_id}/*"
        )
        
        endpoint_url = f"https://{api_id}.execute-api.{self.region}.amazonaws.com/prod/predict"
        print(f"✓ API Gateway created: {endpoint_url}")


def main():
    parser = argparse.ArgumentParser(description="Deploy FrEVL to AWS")
    
    parser.add_argument("service", choices=["ecs", "eks", "sagemaker", "lambda", "all"],
                       help="AWS service to deploy to")
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model",
                       help="Path to model files")
    parser.add_argument("--image", type=str, default="frevl:latest",
                       help="Docker image name")
    parser.add_argument("--region", type=str, default="us-west-2",
                       help="AWS region")
    parser.add_argument("--profile", type=str,
                       help="AWS profile name")
    
    # Service-specific arguments
    parser.add_argument("--cluster", type=str, default="frevl-cluster",
                       help="ECS/EKS cluster name")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge",
                       help="SageMaker instance type")
    parser.add_argument("--gpu", type=int, default=0,
                       help="Number of GPUs for ECS")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = AWSDeployer(region=args.region, profile=args.profile)
    
    # Deploy based on service
    if args.service == "ecs":
        # Push image to ECR
        image_uri = deployer.push_to_ecr("frevl", args.image)
        
        # Deploy to ECS
        deployer.deploy_to_ecs(
            cluster_name=args.cluster,
            service_name="frevl-service",
            task_family="frevl-task",
            image_uri=image_uri,
            gpu=args.gpu
        )
        
    elif args.service == "sagemaker":
        deployer.deploy_to_sagemaker(
            model_name="frevl-model",
            model_path=args.model_path,
            instance_type=args.instance_type
        )
        
    elif args.service == "eks":
        deployer.deploy_to_eks(
            cluster_name=args.cluster
        )
        
    elif args.service == "lambda":
        deployer.deploy_to_lambda(
            function_name="frevl-inference",
            model_path=args.model_path
        )
        
    elif args.service == "all":
        print("Deploying to all services...")
        
        # ECS
        image_uri = deployer.push_to_ecr("frevl", args.image)
        deployer.deploy_to_ecs(
            cluster_name=args.cluster,
            service_name="frevl-service",
            task_family="frevl-task",
            image_uri=image_uri
        )
        
        # SageMaker
        deployer.deploy_to_sagemaker(
            model_name="frevl-model",
            model_path=args.model_path
        )
        
        # Lambda
        deployer.deploy_to_lambda(
            function_name="frevl-inference",
            model_path=args.model_path
        )
    
    print("\n AWS deployment complete!")


if __name__ == "__main__":
    main()
