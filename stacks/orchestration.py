"""Step Functions orchestration for Nova Multi-Turn RL training.

Defines the state machine and Lambda functions that replace setup.sh
with a production-grade, retryable, idempotent workflow.

State machine flow:
  BootstrapEcrImage → DeployRftInfra → StartRewardWorkers → ValidateUploadData → SubmitTraining

The bootstrap step is required because the SDK's ECS infrastructure
(ECSRFTInfrastructure) calls _setup_ecr_image() which runs Docker CLI
commands (docker pull/tag/push) via subprocess. Lambda containers
cannot run Docker. The CodeBuild bootstrap pre-populates the ECR repo
so the SDK's _setup_ecr_image() finds the image and returns early
without any subprocess calls.

Each step has:
  - Retry with exponential backoff (3 attempts)
  - Error catch that routes to a failure state
  - Idempotency checks inside the Lambda handler (or buildspec)
"""
import aws_cdk as cdk
from aws_cdk import (
    Duration,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_codebuild as codebuild,
    aws_ecr_assets as _ecr_assets,
    aws_eks as eks,
    aws_events as events,
    aws_events_targets as targets_eb,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_sns as sns,
    aws_sqs as sqs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from stacks.iam_policies import create_ecs_management_policy, create_cloudwatch_logs_policy


class TrainingOrchestration(Construct):
    """Step Functions orchestration for the training pipeline."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        rft_role: iam.IRole,
        hyperpod_role: iam.IRole,
        bucket_name: str,
        bucket_arn: str,
        cluster_name: str,
        vpc_id: str,
        subnet_ids: str,
        sg_id: str,
        instance_type: str,
        instance_count: int,
        region: str,
        rft_role_arn: str,
        hyperpod_role_arn: str,
        vf_env_id: str = "wordle",
        nova_model: str = "NOVA_LITE_2",
        reward_cluster_arn: str = "",
        reward_cpu: str = "2048",
        reward_memory: str = "4096",
        custom_env_s3_uri: str = "",
        custom_env_id: str = "",
        eks_cluster_name: str = "",
        eks_cluster: "eks.Cluster | None" = None,
        training_method: str = "RFT_MULTITURN_FULL",
        max_steps: str = "10",
        generation_replicas: str = "4",
        global_batch_size: str = "64",
        max_new_tokens: str = "4096",
        max_length: str = "16384",
        training_timeout: str = "1800",
        mlflow_tracking_arn: str = "",
    ) -> None:
        super().__init__(scope, construct_id)

        # Resource naming prefix for SDK-created resources
        # The SDK creates CloudFormation stacks, Lambda functions, SQS queues,
        # DynamoDB tables, and ECS task definitions with this prefix
        # Configured in cdk.json context
        sdk_resource_prefix = self.node.try_get_context("sdk_resource_prefix") or "nrl"

        # Extract ECS cluster name from ARN for IAM policy scoping
        # ARN format: arn:aws:ecs:REGION:ACCOUNT:cluster/CLUSTER_NAME
        reward_cluster_name = reward_cluster_arn.split("/")[-1] if reward_cluster_arn else ""

        # --- Customer Managed Policies ---
        # Reusable policies that can be attached to multiple roles
        
        # Policy 1: CloudWatch Logs for Lambda execution
        cloudwatch_logs_policy = create_cloudwatch_logs_policy(
            self, "LambdaCloudWatchLogsPolicy",
            region=region, log_group_prefix="/aws/lambda/*",
        )

        # Policy 2: CloudFormation management for SDK stack creation
        # SAM CLI requires additional permissions beyond basic stack operations
        cloudformation_management_policy = iam.ManagedPolicy(
            self,
            "CloudFormationManagementPolicy",
            description="CloudFormation permissions for SDK stack management",
            statements=[
                iam.PolicyStatement(
                    sid="CloudFormationManagement",
                    actions=[
                        "cloudformation:CreateStack",
                        "cloudformation:UpdateStack",
                        "cloudformation:DeleteStack",
                        "cloudformation:DescribeStacks",
                        "cloudformation:DescribeStackEvents",
                        "cloudformation:DescribeStackResources",
                        "cloudformation:GetTemplate",
                        "cloudformation:ListStackResources",
                        "cloudformation:ValidateTemplate",
                        "cloudformation:CreateChangeSet",
                        "cloudformation:DescribeChangeSet",
                        "cloudformation:ExecuteChangeSet",
                        "cloudformation:DeleteChangeSet",
                        "cloudformation:ListChangeSets",
                        "cloudformation:GetTemplateSummary",
                    ],
                    resources=[
                        f"arn:aws:cloudformation:{region}:{cdk.Aws.ACCOUNT_ID}:stack/{sdk_resource_prefix}-*",
                        f"arn:aws:cloudformation:{region}:{cdk.Aws.ACCOUNT_ID}:stackset/*",
                    ],
                )
            ],
        )

        # Suppress AwsSolutions-IAM5 for CloudFormation management wildcards
        NagSuppressions.add_resource_suppressions(
            cloudformation_management_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": f"SDK creates CloudFormation stacks with {sdk_resource_prefix}- prefix. stackset/* is required for SAM CLI operations.",
                    "appliesTo": [
                        f"Resource::arn:aws:cloudformation:{region}:<AWS::AccountId>:stack/{sdk_resource_prefix}-*",
                        f"Resource::arn:aws:cloudformation:{region}:<AWS::AccountId>:stackset/*",
                    ],
                }
            ],
        )

        # Policy 3: Lambda/SQS/DynamoDB management for SDK infrastructure
        sdk_infrastructure_policy = iam.ManagedPolicy(
            self,
            "SdkInfrastructurePolicy",
            description="Lambda, SQS, and DynamoDB management for SDK infrastructure",
            statements=[
                iam.PolicyStatement(
                    sid="LambdaManagement",
                    actions=[
                        "lambda:CreateFunction",
                        "lambda:DeleteFunction",
                        "lambda:GetFunction",
                        "lambda:InvokeFunction",
                        "lambda:UpdateFunctionCode",
                        "lambda:UpdateFunctionConfiguration",
                        "lambda:AddPermission",
                        "lambda:RemovePermission",
                        "lambda:CreateFunctionUrlConfig",
                        "lambda:GetFunctionUrlConfig",
                        "lambda:DeleteFunctionUrlConfig",
                    ],
                    resources=[
                        f"arn:aws:lambda:{region}:{cdk.Aws.ACCOUNT_ID}:function:{sdk_resource_prefix}-*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="SqsManagement",
                    actions=[
                        "sqs:CreateQueue",
                        "sqs:DeleteQueue",
                        "sqs:GetQueueAttributes",
                        "sqs:SetQueueAttributes",
                        "sqs:SendMessage",
                        "sqs:GetQueueUrl",
                        "sqs:TagQueue",
                    ],
                    resources=[
                        f"arn:aws:sqs:{region}:{cdk.Aws.ACCOUNT_ID}:{sdk_resource_prefix}-*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="DynamoDbManagement",
                    actions=[
                        "dynamodb:CreateTable",
                        "dynamodb:DeleteTable",
                        "dynamodb:DescribeTable",
                        "dynamodb:PutItem",
                        "dynamodb:GetItem",
                        "dynamodb:TagResource",
                    ],
                    resources=[
                        f"arn:aws:dynamodb:{region}:{cdk.Aws.ACCOUNT_ID}:table/{sdk_resource_prefix}-*",
                    ],
                ),
            ],
        )

        # Suppress AwsSolutions-IAM5 for SDK infrastructure wildcards
        NagSuppressions.add_resource_suppressions(
            sdk_infrastructure_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": f"SDK creates Lambda functions, SQS queues, and DynamoDB tables with {sdk_resource_prefix}- prefix. Wildcards are scoped to this prefix.",
                    "appliesTo": [
                        f"Resource::arn:aws:lambda:{region}:<AWS::AccountId>:function:{sdk_resource_prefix}-*",
                        f"Resource::arn:aws:sqs:{region}:<AWS::AccountId>:{sdk_resource_prefix}-*",
                        f"Resource::arn:aws:dynamodb:{region}:<AWS::AccountId>:table/{sdk_resource_prefix}-*",
                    ],
                }
            ],
        )

        # Policy 4: IAM role management for SDK-created resources
        iam_role_management_policy = iam.ManagedPolicy(
            self,
            "IamRoleManagementPolicy",
            description="IAM role and policy management for SDK-created resources",
            statements=[
                iam.PolicyStatement(
                    sid="IamRoleManagement",
                    actions=[
                        "iam:CreateRole",
                        "iam:DeleteRole",
                        "iam:GetRole",
                        "iam:AttachRolePolicy",
                        "iam:DetachRolePolicy",
                        "iam:PutRolePolicy",
                        "iam:DeleteRolePolicy",
                        "iam:TagRole",
                        "iam:CreatePolicy",
                        "iam:ListPolicies",
                        "iam:SimulatePrincipalPolicy",
                    ],
                    resources=[
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:role/*nova*",
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:role/*rft*",
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:role/*Nova*",
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:role/*RFT*",
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:policy/*nova*",
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:policy/*rft*",
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:policy/*Nova*",
                        f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:policy/*RFT*",
                    ],
                )
            ],
        )
        
        # Suppress AwsSolutions-IAM5 for IAM role management wildcards
        # The SDK dynamically creates IAM roles and policies with unpredictable names
        # Pattern matching on *nova*, *rft*, *Nova*, *RFT* is the tightest constraint possible
        NagSuppressions.add_resource_suppressions(
            iam_role_management_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK dynamically creates IAM roles and policies with unpredictable names. Wildcards are scoped to nova/rft patterns which is the tightest constraint possible for SDK-managed resources.",
                    "appliesTo": [
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*nova*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*rft*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*Nova*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*RFT*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:policy/*nova*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:policy/*rft*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:policy/*Nova*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:policy/*RFT*",
                    ],
                }
            ],
        )

        # Policy 5: IAM PassRole for SDK to assign roles
        # This policy allows Lambda functions to pass the rft_role to SDK-created resources
        iam_pass_role_policy = iam.ManagedPolicy(
            self,
            "IamPassRolePolicy",
            description="IAM PassRole permissions for SDK to assign roles to Lambda/ECS",
            statements=[
                iam.PolicyStatement(
                    sid="IamPassRole",
                    actions=["iam:PassRole"],
                    resources=[rft_role.role_arn],
                )
            ],
        )

        # Policy 6: ECR read permissions for SDK
        ecr_read_policy = iam.ManagedPolicy(
            self,
            "EcrReadPolicy",
            description="ECR read permissions for SDK image checks",
            statements=[
                iam.PolicyStatement(
                    sid="EcrRead",
                    actions=[
                        "ecr:DescribeRepositories",
                        "ecr:DescribeImages",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                    ],
                    resources=[
                        f"arn:aws:ecr:{region}:{cdk.Aws.ACCOUNT_ID}:repository/nova-rft-base",
                    ],
                ),
                iam.PolicyStatement(
                    sid="EcrGetAuthorizationToken",
                    actions=["ecr:GetAuthorizationToken"],
                    resources=["*"],  # GetAuthorizationToken doesn't support resource-level permissions
                ),
            ],
        )
        
        # Suppress AwsSolutions-IAM5 for ECR GetAuthorizationToken wildcard
        # GetAuthorizationToken is a global ECR action that doesn't support resource-level permissions
        NagSuppressions.add_resource_suppressions(
            ecr_read_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ecr:GetAuthorizationToken does not support resource-level permissions and requires wildcard. This is an AWS service limitation documented in the ECR API reference.",
                }
            ],
        )

        # Policy 7: ECS management for reward workers
        ecs_management_policy = create_ecs_management_policy(
            self, "EcsManagementPolicy",
        )

        # Policy 8: S3 access for training data
        s3_data_access_policy = iam.ManagedPolicy(
            self,
            "S3DataAccessPolicy",
            description="S3 permissions for training data validation and upload",
            statements=[
                iam.PolicyStatement(
                    sid="S3DataAccess",
                    actions=[
                        "s3:PutObject",
                        "s3:GetObject",
                        "s3:ListBucket",
                        "s3:HeadObject",
                    ],
                    resources=[
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                )
            ],
        )
        
        # Suppress AwsSolutions-IAM5 for S3 bucket object wildcard
        # S3 object operations require /* wildcard to access all objects in the bucket
        NagSuppressions.add_resource_suppressions(
            s3_data_access_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 object operations (GetObject, PutObject) require /* wildcard to access all training data files in the bucket. This is standard S3 access pattern.",
                    "appliesTo": [f"Resource::arn:aws:s3:::<TrainingBucketEB7BB5C9>/*"],
                }
            ],
        )

        # --- Role 1: Deploy RFT Infrastructure Lambda ---
        # Creates CloudFormation stack with Lambda, SQS, DynamoDB via SDK
        deploy_rft_role = iam.Role(
            self,
            "DeployRftInfraRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for deploy_rft_infra Lambda - creates SDK CloudFormation stack",
            managed_policies=[
                cloudwatch_logs_policy,
                cloudformation_management_policy,
                sdk_infrastructure_policy,
                iam_role_management_policy,
                iam_pass_role_policy,
                ecr_read_policy,
                ecs_management_policy
            ],
        )

        # --- Role 2: Start Reward Workers Lambda ---
        # Starts ECS Fargate tasks for reward environment
        start_workers_role = iam.Role(
            self,
            "StartRewardWorkersRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for start_reward_workers Lambda - starts ECS Fargate tasks",
            managed_policies=[
                cloudwatch_logs_policy,
                cloudformation_management_policy,
                ecs_management_policy,
                iam_role_management_policy,
                iam_pass_role_policy,
                ecr_read_policy,
                s3_data_access_policy,
            ],
        )

        # --- Role 3: Validate Upload Data Lambda ---
        # Validates JSONL and uploads to S3
        validate_data_role = iam.Role(
            self,
            "ValidateUploadDataRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for validate_upload_data Lambda - validates JSONL and uploads to S3",
            managed_policies=[
                cloudwatch_logs_policy,
                s3_data_access_policy,
            ],
        )

        # --- Lambda Functions (one per step) ---
        # Docker-based packaging: the SDK and its dependencies are installed
        # in the container image. Each function shares the same Dockerfile
        # with a build arg selecting the handler directory.
        #
        # All 3 functions use the same base image (only the FUNCTION_DIR
        # build arg differs). CDK's asset hashing means Docker layer caching
        # keeps subsequent builds fast, but the first deploy builds 3 images
        # sequentially. If this becomes a bottleneck, consider a single shared
        # ECR image with an entrypoint that selects the handler at runtime.
        lambda_timeout = Duration.minutes(15)
        lambda_memory = 512
        lambda_env = {
            "POWERTOOLS_SERVICE_NAME": "nova-rl-orchestration",
            "SDK_RESOURCE_PREFIX": sdk_resource_prefix,
        }

        deploy_rft_fn = _lambda.DockerImageFunction(
            self,
            "DeployRftInfraFn",
            code=_lambda.DockerImageCode.from_image_asset(
                "lambdas",
                build_args={"FUNCTION_DIR": "deploy_rft_infra"},
                cmd=["handler.handler"],
                platform=_ecr_assets.Platform.LINUX_AMD64,
            ),
            description="Step 1: Deploy RFT multi-turn infrastructure via SDK",
            timeout=lambda_timeout,
            memory_size=lambda_memory,
            role=deploy_rft_role,
            environment=lambda_env,
        )

        start_workers_fn = _lambda.DockerImageFunction(
            self,
            "StartRewardWorkersFn",
            code=_lambda.DockerImageCode.from_image_asset(
                "lambdas",
                build_args={"FUNCTION_DIR": "start_reward_workers"},
                cmd=["handler.handler"],
                platform=_ecr_assets.Platform.LINUX_AMD64,
            ),
            description="Step 2: Start reward workers on ECS Fargate",
            timeout=lambda_timeout,
            memory_size=lambda_memory,
            role=start_workers_role,
            environment=lambda_env,
        )

        validate_data_fn = _lambda.DockerImageFunction(
            self,
            "ValidateUploadDataFn",
            code=_lambda.DockerImageCode.from_image_asset(
                "lambdas",
                build_args={"FUNCTION_DIR": "validate_upload_data"},
                cmd=["handler.handler"],
                platform=_ecr_assets.Platform.LINUX_AMD64,
            ),
            description="Step 3: Validate and upload training data to S3",
            timeout=lambda_timeout,
            memory_size=lambda_memory,
            role=validate_data_role,
            environment=lambda_env,
        )

        # --- Customer Managed Policies for CodeBuild ---
        # These policies must be defined before the CodeBuild project that uses them
        
        forge_s3_bucket = "nova-forge-c7363-206080352451-us-east-1"

        # Policy 10: SageMaker training job management for CodeBuild
        sagemaker_training_policy = iam.ManagedPolicy(
            self,
            "SageMakerTrainingPolicy",
            description="SageMaker permissions for training job submission",
            statements=[
                iam.PolicyStatement(
                    sid="SageMakerTrainingJobs",
                    actions=[
                        "sagemaker:CreateTrainingJob",
                        "sagemaker:DescribeTrainingJob",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{region}:{cdk.Aws.ACCOUNT_ID}:training-job/nova-rl-*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="SageMakerClusterAccess",
                    actions=[
                        "sagemaker:DescribeCluster",
                        "sagemaker:DescribeClusterNode",
                        "sagemaker:ListClusterNodes",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{region}:{cdk.Aws.ACCOUNT_ID}:cluster/*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="SageMakerHubContent",
                    actions=[
                        "sagemaker:DescribeHubContent",
                        "sagemaker:DescribeHub",
                    ],
                    resources=["*"],
                ),
                # The SDK runs iam:SimulatePrincipalPolicy as a pre-flight
                # check WITHOUT resource ARNs, so resource-scoped policies
                # appear as "denied". This wildcard statement satisfies the
                # simulator for actions already scoped above.
                iam.PolicyStatement(
                    sid="SdkSimulatorPreflightCheck",
                    actions=[
                        "sagemaker:DescribeCluster",
                        "sagemaker:ListClusters",
                        "eks:DescribeCluster",
                        "eks:ListAddons",
                        "s3:CreateBucket",
                    ],
                    resources=["*"],
                ),
            ],
        )

        # Suppress AwsSolutions-IAM5 for SageMaker training policy wildcards
        NagSuppressions.add_resource_suppressions(
            sagemaker_training_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SageMaker training jobs are created with nova-rl- prefix. cluster/* is required for HyperPod cluster discovery. Hub content and preflight check require wildcard for SDK compatibility.",
                    "appliesTo": [
                        f"Resource::arn:aws:sagemaker:{region}:<AWS::AccountId>:training-job/nova-rl-*",
                        f"Resource::arn:aws:sagemaker:{region}:<AWS::AccountId>:cluster/*",
                        "Resource::*",
                    ],
                }
            ],
        )

        # Policy 11: EKS access for HyperPod CLI
        eks_access_policy = iam.ManagedPolicy(
            self,
            "EksAccessPolicy",
            description="EKS permissions for HyperPod CLI kubectl operations",
            statements=[
                iam.PolicyStatement(
                    sid="EksAccess",
                    actions=[
                        "eks:DescribeCluster",
                        "eks:ListClusters",
                        "eks:AccessKubernetesApi",
                        "eks:ListAddons",
                    ],
                    resources=[
                        f"arn:aws:eks:{region}:{cdk.Aws.ACCOUNT_ID}:cluster/*",
                    ],
                )
            ],
        )

        # Suppress AwsSolutions-IAM5 for EKS access wildcards
        NagSuppressions.add_resource_suppressions(
            eks_access_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "EKS cluster/* wildcard is required because HyperPod CLI needs to discover and connect to clusters dynamically. Cluster name is not known at deploy time when targeting existing clusters.",
                    "appliesTo": [
                        f"Resource::arn:aws:eks:{region}:<AWS::AccountId>:cluster/*",
                    ],
                }
            ],
        )

        # Policy 12: S3 read access for training scripts and SDK resources
        s3_training_read_policy = iam.ManagedPolicy(
            self,
            "S3TrainingReadPolicy",
            description="S3 read permissions for training data and SDK resources",
            statements=[
                iam.PolicyStatement(
                    sid="S3TrainingBucketAccess",
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket",
                    ],
                    resources=[
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="S3ForgeBucketRead",
                    actions=[
                        "s3:GetObject",
                        "s3:ListBucket",
                    ],
                    resources=[
                        f"arn:aws:s3:::{forge_s3_bucket}",
                        f"arn:aws:s3:::{forge_s3_bucket}/*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="S3JumpStartCacheRead",
                    actions=[
                        "s3:GetObject",
                        "s3:ListBucket",
                    ],
                    resources=[
                        "arn:aws:s3:::jumpstart-cache-prod-*",
                        "arn:aws:s3:::jumpstart-cache-prod-*/*",
                    ],
                ),
            ],
        )

        # Suppress AwsSolutions-IAM5 for S3 training read wildcards
        NagSuppressions.add_resource_suppressions(
            s3_training_read_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 object operations require /* wildcard for bucket access. Training bucket, Forge bucket, and JumpStart cache all contain dynamic object keys that cannot be enumerated at deploy time.",
                    "appliesTo": [
                        f"Resource::arn:aws:s3:::<TrainingBucketEB7BB5C9>/*",
                        f"Resource::arn:aws:s3:::{forge_s3_bucket}/*",
                        "Resource::arn:aws:s3:::jumpstart-cache-prod-*",
                        "Resource::arn:aws:s3:::jumpstart-cache-prod-*/*",
                    ],
                }
            ],
        )

        # Policy 13: SDK infrastructure read for CodeBuild
        sdk_infrastructure_read_policy = iam.ManagedPolicy(
            self,
            "SdkInfrastructureReadPolicy",
            description="Read permissions for SDK-created infrastructure resources",
            statements=[
                iam.PolicyStatement(
                    sid="CloudFormationRead",
                    actions=[
                        "cloudformation:DescribeStacks",
                        "cloudformation:DescribeStackResources",
                        "cloudformation:ListStackResources",
                    ],
                    resources=[
                        f"arn:aws:cloudformation:{region}:{cdk.Aws.ACCOUNT_ID}:stack/{sdk_resource_prefix}-*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="EcsRead",
                    actions=[
                        "ecs:DescribeClusters",
                        "ecs:DescribeTasks",
                        "ecs:ListTasks",
                    ],
                    resources=[
                        f"arn:aws:ecs:{region}:{cdk.Aws.ACCOUNT_ID}:cluster/{reward_cluster_name}",
                        f"arn:aws:ecs:{region}:{cdk.Aws.ACCOUNT_ID}:task-definition/{sdk_resource_prefix}-*",
                        f"arn:aws:ecs:{region}:{cdk.Aws.ACCOUNT_ID}:task/{reward_cluster_name}/*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="EcsDescribeTaskDefinition",
                    actions=["ecs:DescribeTaskDefinition"],
                    resources=["*"],  # DescribeTaskDefinition doesn't support resource-level permissions
                ),
                iam.PolicyStatement(
                    sid="LambdaRead",
                    actions=[
                        "lambda:GetFunction",
                        "lambda:GetFunctionUrlConfig",
                    ],
                    resources=[
                        f"arn:aws:lambda:{region}:{cdk.Aws.ACCOUNT_ID}:function:{sdk_resource_prefix}-*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="SqsRead",
                    actions=["sqs:GetQueueAttributes", "sqs:GetQueueUrl"],
                    resources=[
                        f"arn:aws:sqs:{region}:{cdk.Aws.ACCOUNT_ID}:{sdk_resource_prefix}-*",
                    ],
                ),
                iam.PolicyStatement(
                    sid="DynamoDbRead",
                    actions=["dynamodb:DescribeTable"],
                    resources=[
                        f"arn:aws:dynamodb:{region}:{cdk.Aws.ACCOUNT_ID}:table/{sdk_resource_prefix}-*",
                    ],
                ),
            ],
        )

        # Suppress AwsSolutions-IAM5 for SDK infrastructure read wildcards
        NagSuppressions.add_resource_suppressions(
            sdk_infrastructure_read_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK infrastructure read requires wildcards for CloudFormation stacks, ECS tasks, Lambda functions, SQS queues, and DynamoDB tables created by the SDK with dynamic names. DescribeTaskDefinition requires Resource::* (AWS service limitation).",
                    "appliesTo": [
                        f"Resource::arn:aws:cloudformation:{region}:<AWS::AccountId>:stack/{sdk_resource_prefix}-*",
                        f"Resource::arn:aws:ecs:{region}:<AWS::AccountId>:task-definition/{sdk_resource_prefix}-*",
                        f"Resource::arn:aws:ecs:{region}:<AWS::AccountId>:task/{reward_cluster_name}/*",
                        f"Resource::arn:aws:lambda:{region}:<AWS::AccountId>:function:{sdk_resource_prefix}-*",
                        f"Resource::arn:aws:sqs:{region}:<AWS::AccountId>:{sdk_resource_prefix}-*",
                        f"Resource::arn:aws:dynamodb:{region}:<AWS::AccountId>:table/{sdk_resource_prefix}-*",
                        "Resource::*",
                    ],
                }
            ],
        )

        # Policy 14: IAM PassRole for training job execution
        training_pass_role_policy = iam.ManagedPolicy(
            self,
            "TrainingPassRolePolicy",
            description="IAM PassRole for SageMaker training job execution",
            statements=[
                iam.PolicyStatement(
                    sid="PassRoleForTraining",
                    actions=["iam:PassRole"],
                    resources=[
                        rft_role.role_arn,
                        hyperpod_role.role_arn,
                    ],
                    conditions={
                        "StringEquals": {
                            "iam:PassedToService": "sagemaker.amazonaws.com"
                        }
                    },
                )
            ],
        )

        # --- CodeBuild Step 4: Submit Training ---
        # SMHPRuntimeManager requires the hyperpod CLI (kubectl + helm) which
        # cannot run in Lambda. CodeBuild provides a full Linux environment
        # where we install the CLI and submit the training job.

        # KMS key for CodeBuild project encryption (AwsSolutions-CB4)
        codebuild_kms_key = kms.Key(
            self,
            "CodeBuildEncryptionKey",
            description="KMS key for encrypting CodeBuild project artifacts and logs",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        submit_training_project = codebuild.Project(
            self,
            "SubmitTrainingProject",
            project_name="nova-rl-submit-training",
            description="Step 4: Install HyperPod CLI and submit training job to HyperPod",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.SMALL,
            ),
            encryption_key=codebuild_kms_key,
            timeout=Duration.minutes(30),
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "runtime-versions": {"python": "3.12"},
                        "commands": [
                            # Install kubectl
                            "curl -LO https://dl.k8s.io/release/v1.31.0/bin/linux/amd64/kubectl",
                            "chmod +x kubectl && mv kubectl /usr/local/bin/kubectl",
                            # Install helm
                            "curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3",
                            "chmod 700 get_helm.sh && ./get_helm.sh && rm -f get_helm.sh",
                            # Install HyperPod CLI from Forge S3 FIRST (stricter version pins)
                            f"mkdir -p /tmp/hyperpod-cli && cd /tmp/hyperpod-cli && aws s3 cp s3://{forge_s3_bucket}/v1/ ./ --recursive && mkdir -p src/hyperpod_cli/sagemaker_hyperpod_recipes/launcher/nemo && git clone https://github.com/NVIDIA/NeMo-Framework-Launcher.git src/hyperpod_cli/sagemaker_hyperpod_recipes/launcher/nemo/nemo_framework_launcher --recursive --depth 1 && pip install . && hyperpod --help",
                            # Install Nova SDK without overwriting CLI's pinned deps
                            "pip install --no-deps amzn-nova-forge==1.3.16",
                            "pip install sagemaker==2.254.1 numpy",
                            # Suppress Python dependency warnings that the HyperPod CLI
                            # emits to stderr. The SDK's SMHPRuntimeManager treats any
                            # stderr output as a connection failure (even on exit code 0).
                            # The warning is caused by chardet 6.x (from CLI) being newer
                            # than what requests 2.32.5 officially supports — harmless but
                            # fatal to the SDK's stderr check.
                            "export PYTHONWARNINGS=ignore",
                            # NeMo launcher module must be on PYTHONPATH for hyperpod start-job
                            "export PYTHONPATH=/tmp/hyperpod-cli/src/hyperpod_cli/sagemaker_hyperpod_recipes/launcher/nemo/nemo_framework_launcher/launcher_scripts:${PYTHONPATH:-}",
                        ],
                    },
                    "build": {
                        "commands": [
                            # Configure EKS access — required before hyperpod connect-cluster
                            f"aws eks update-kubeconfig --name {eks_cluster_name} --region {region}",
                            # Connect HyperPod cluster to kubectl context
                            f"hyperpod connect-cluster --cluster-name {cluster_name}",
                            # Download and run the submit script
                            f"aws s3 cp s3://$BUCKET_NAME/codebuild-scripts/codebuild_submit.py /tmp/codebuild_submit.py",
                            "cd /tmp && python codebuild_submit.py",
                        ],
                    },
                },
            }),
        )

        # Attach customer managed policies to CodeBuild role
        submit_training_project.role.add_managed_policy(sagemaker_training_policy)
        submit_training_project.role.add_managed_policy(eks_access_policy)
        submit_training_project.role.add_managed_policy(s3_training_read_policy)
        submit_training_project.role.add_managed_policy(sdk_infrastructure_read_policy)
        submit_training_project.role.add_managed_policy(training_pass_role_policy)
        submit_training_project.role.add_managed_policy(ecr_read_policy)
        submit_training_project.role.add_managed_policy(iam_role_management_policy)

        # Suppress AwsSolutions-IAM5 for CodeBuild auto-generated default policy
        # CDK creates a default policy for CodeBuild with log group, report group,
        # and KMS permissions using wildcards that cannot be customized
        NagSuppressions.add_resource_suppressions(
            submit_training_project,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CDK auto-generates CodeBuild default policy with wildcard permissions for log groups (log-group:/aws/codebuild/*), report groups (report-group:nova-rl-*), and KMS operations (kms:ReEncrypt*, kms:GenerateDataKey*). These are CDK-managed and cannot be customized.",
                }
            ],
            apply_to_children=True,
        )

        # Grant the CodeBuild submit training role access to the EKS cluster
        # so `aws eks update-kubeconfig` and `kubectl` commands work.
        # Without this, CodeBuild cannot authenticate to the EKS API server.
        if eks_cluster:
            eks_cluster.aws_auth.add_role_mapping(
                submit_training_project.role,
                groups=["system:masters"],
                username="codebuild-submit-training",
            )

        # --- CodeBuild bootstrap: pre-populate ECR image for SDK ---
        # The SDK's ECSRFTInfrastructure._setup_ecr_image() runs Docker CLI
        # commands (docker pull/tag/push) via subprocess to push python:3.12-slim
        # to a private ECR repo named "nova-rft-base". Lambda cannot run Docker.
        # This CodeBuild step does the same thing in an environment that has Docker,
        # so by the time the Lambda handlers call _setup_ecr_image(), the image
        # already exists and the SDK skips the subprocess calls entirely.
        ecr_repo_name = "nova-rft-base"
        sdk_base_image = "public.ecr.aws/docker/library/python:3.12-slim"

        bootstrap_project = codebuild.Project(
            self,
            "EcrBootstrapProject",
            project_name="nova-rl-ecr-bootstrap",
            description="Pre-populates ECR image required by Nova SDK ECS infrastructure",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                privileged=True,  # Required for Docker commands
                compute_type=codebuild.ComputeType.SMALL,
            ),
            encryption_key=codebuild_kms_key,
            timeout=Duration.minutes(10),
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "pre_build": {
                        "commands": [
                            # Check if image already exists — skip if so (idempotent)
                            # Each command must be self-contained; CodeBuild runs them independently.
                            f'REPO_URI=$(aws ecr describe-repositories --repository-names {ecr_repo_name} --query "repositories[0].repositoryUri" --output text 2>/dev/null || echo "NONE")',
                            f'if [ "$REPO_URI" != "NONE" ]; then IMAGE_EXISTS=$(aws ecr describe-images --repository-name {ecr_repo_name} --image-ids imageTag=latest --query "imageDetails[0].imageTags" --output text 2>/dev/null || echo "NONE"); if [ "$IMAGE_EXISTS" != "NONE" ]; then echo "Image already exists in ECR. Skipping bootstrap."; exit 0; fi; fi',
                        ],
                    },
                    "build": {
                        "commands": [
                            # Create ECR repo if it doesn't exist
                            f'aws ecr describe-repositories --repository-names {ecr_repo_name} 2>/dev/null || aws ecr create-repository --repository-name {ecr_repo_name}',
                            f'REPO_URI=$(aws ecr describe-repositories --repository-names {ecr_repo_name} --query "repositories[0].repositoryUri" --output text)',
                            # Login to ECR
                            'aws ecr get-login-password | docker login --username AWS --password-stdin $REPO_URI',
                            # Pull, tag, push
                            f'docker pull {sdk_base_image}',
                            f'docker tag {sdk_base_image} $REPO_URI:latest',
                            'docker push $REPO_URI:latest',
                            'echo "ECR bootstrap complete: $REPO_URI:latest"',
                        ],
                    },
                },
            }),
        )

        # Grant CodeBuild permissions to manage ECR
        bootstrap_project.add_to_role_policy(
            iam.PolicyStatement(
                sid="EcrBootstrap",
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:CreateRepository",
                    "ecr:DescribeRepositories",
                    "ecr:DescribeImages",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:PutImage",
                ],
                resources=["*"],
            )
        )
        
        # Suppress AwsSolutions-IAM5 for ECR bootstrap wildcard permissions
        # This is a one-time bootstrap operation that needs broad ECR permissions
        # to create and populate the repository before the SDK can use it
        NagSuppressions.add_resource_suppressions(
            bootstrap_project,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ECR bootstrap requires wildcard permissions for one-time repository creation and image push. GetAuthorizationToken and CreateRepository do not support resource-level permissions.",
                }
            ],
            apply_to_children=True,
        )

        # --- Step Functions tasks ---
        bootstrap_ecr_task = tasks.CodeBuildStartBuild(
            self,
            "BootstrapEcrImage",
            project=bootstrap_project,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            comment="Pre-populate ECR image so SDK skips Docker subprocess in Lambda",
            result_path="$.bootstrap_result",
        )
        bootstrap_ecr_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(30),
            max_attempts=2,
            backoff_rate=2.0,
        )

        deploy_rft_task = tasks.LambdaInvoke(
            self,
            "DeployRftInfra",
            lambda_function=deploy_rft_fn,
            output_path="$.Payload",
            comment="Deploy RFT multi-turn infrastructure (Lambda, SQS, DynamoDB)",
        )
        deploy_rft_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(30),
            max_attempts=3,
            backoff_rate=2.0,
        )

        start_workers_task = tasks.LambdaInvoke(
            self,
            "StartRewardWorkers",
            lambda_function=start_workers_fn,
            output_path="$.Payload",
            comment="Start reward workers on ECS Fargate",
        )
        start_workers_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(30),
            max_attempts=3,
            backoff_rate=2.0,
        )

        validate_data_task = tasks.LambdaInvoke(
            self,
            "ValidateUploadData",
            lambda_function=validate_data_fn,
            output_path="$.Payload",
            comment="Validate and upload training data to S3",
        )
        validate_data_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(30),
            max_attempts=3,
            backoff_rate=2.0,
        )

        submit_training_task = tasks.CodeBuildStartBuild(
            self,
            "SubmitTraining",
            project=submit_training_project,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            comment="Submit training job to HyperPod via CodeBuild (requires hyperpod CLI)",
            environment_variables_override={
                "PIPELINE_EVENT": codebuild.BuildEnvironmentVariable(
                    value=sfn.JsonPath.string_at("States.JsonToString($)"),
                ),
                "BUCKET_NAME": codebuild.BuildEnvironmentVariable(
                    value=bucket_name,
                ),
                "SDK_RESOURCE_PREFIX": codebuild.BuildEnvironmentVariable(
                    value=sdk_resource_prefix,
                ),
            },
            result_path="$.submit_result",
        )
        submit_training_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(60),
            max_attempts=2,
            backoff_rate=2.0,
        )

        # --- Failure state ---
        pipeline_failed = sfn.Fail(
            self,
            "PipelineFailed",
            cause="Training pipeline step failed after retries",
            error="PIPELINE_FAILURE",
        )

        # --- Success state ---
        pipeline_succeeded = sfn.Succeed(
            self,
            "PipelineSucceeded",
            comment="Training pipeline completed successfully",
        )

        # --- Error catch for each task ---
        for task in [bootstrap_ecr_task, deploy_rft_task, start_workers_task, validate_data_task, submit_training_task]:
            task.add_catch(pipeline_failed, errors=["States.ALL"])

        # --- Chain the tasks ---
        chain = (
            bootstrap_ecr_task
            .next(deploy_rft_task)
            .next(start_workers_task)
            .next(validate_data_task)
            .next(submit_training_task)
            .next(pipeline_succeeded)
        )

        # --- Log group for state machine execution ---
        orchestration_log_key = kms.Key(self, "OrchestrationLogKey",
            description="KMS key for orchestration CloudWatch log encryption",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        orchestration_log_key.grant_encrypt_decrypt(
            iam.ServicePrincipal(f"logs.{cdk.Aws.REGION}.amazonaws.com")
        )

        log_group = logs.LogGroup(
            self,
            "OrchestrationLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            encryption_key=orchestration_log_key,
        )

        # --- State Machine ---
        self.state_machine = sfn.StateMachine(
            self,
            "TrainingPipeline",
            state_machine_name="nova-rl-training-pipeline",
            definition_body=sfn.DefinitionBody.from_chainable(chain),
            timeout=Duration.hours(2),
            tracing_enabled=True,
            logs=sfn.LogOptions(
                destination=log_group,
                level=sfn.LogLevel.ALL,
            ),
        )

        # ============================================================
        # Pipeline failure alerting: SNS topic + CloudWatch alarm
        # ============================================================
        sns_kms_key = kms.Key(
            self,
            "SnsEncryptionKey",
            description="KMS key for SNS topic encryption",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        self.alert_topic = sns.Topic(
            self,
            "PipelineAlertTopic",
            display_name="Nova RL Pipeline Failure Alerts",
            master_key=sns_kms_key,
        )

        # Enforce SSL/TLS for all SNS publish operations (AwsSolutions-SNS3)
        self.alert_topic.add_to_resource_policy(
            iam.PolicyStatement(
                sid="EnforceSSLOnly",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["sns:Publish"],
                resources=[self.alert_topic.topic_arn],
                conditions={
                    "Bool": {"aws:SecureTransport": "false"}
                },
            )
        )

        # Expose the topic ARN as a stack output so operators can subscribe
        # (email, Slack webhook, PagerDuty, etc.) after deployment.
        cdk.CfnOutput(
            scope,
            "AlertTopicArn",
            value=self.alert_topic.topic_arn,
            description="SNS topic ARN for pipeline failure alerts — subscribe to receive notifications",
        )

        cloudwatch.Alarm(
            self,
            "PipelineFailureAlarm",
            metric=self.state_machine.metric_failed(
                period=Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            alarm_description="Fires when any Step Functions execution reaches the PipelineFailed state",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        ).add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # ============================================================
        # Lambda function error and throttle alarms
        # ============================================================
        for fn_name, fn_ref in [
            ("DeployRftInfra", deploy_rft_fn),
            ("StartRewardWorkers", start_workers_fn),
            ("ValidateUploadData", validate_data_fn),
        ]:
            cloudwatch.Alarm(
                self,
                f"{fn_name}ErrorAlarm",
                metric=fn_ref.metric_errors(period=Duration.minutes(5), statistic="Sum"),
                threshold=1,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                alarm_description=f"{fn_name} Lambda error alarm",
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ).add_alarm_action(cw_actions.SnsAction(self.alert_topic))

            cloudwatch.Alarm(
                self,
                f"{fn_name}ThrottleAlarm",
                metric=fn_ref.metric_throttles(period=Duration.minutes(5), statistic="Sum"),
                threshold=1,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                alarm_description=f"{fn_name} Lambda throttle alarm",
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ).add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # ============================================================
        # EventBridge trigger: start pipeline when training data lands in S3
        # ============================================================
        # Fires when any .jsonl file is created under the training-data/ prefix.
        # A small Lambda transforms the S3 event into the Step Functions input
        # format, injecting all the infrastructure references the pipeline needs.

        # Policy 9: Step Functions execution for S3 trigger Lambda
        step_functions_execution_policy = iam.ManagedPolicy(
            self,
            "StepFunctionsExecutionPolicy",
            description="Step Functions execution permissions for S3 trigger Lambda",
            statements=[
                iam.PolicyStatement(
                    sid="StepFunctionsExecution",
                    actions=["states:StartExecution"],
                    resources=[self.state_machine.state_machine_arn],
                )
            ],
        )

        # Create a role for the S3 trigger Lambda with customer managed policies
        trigger_fn_role = iam.Role(
            self,
            "S3TriggerFnRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for S3 trigger Lambda",
            managed_policies=[
                cloudwatch_logs_policy,
                step_functions_execution_policy,
            ],
        )

        trigger_fn = _lambda.Function(
            self,
            "S3TriggerFn",
            runtime=_lambda.Runtime.PYTHON_3_13,  # AwsSolutions-L1: Use latest runtime
            handler="handler.handler",
            timeout=Duration.seconds(30),
            memory_size=128,
            description="Transforms S3 events into Step Functions pipeline input",
            role=trigger_fn_role,
            environment={
                "CLUSTER_NAME": cluster_name,
                "BUCKET_NAME": bucket_name,
                "RFT_ROLE_ARN": rft_role_arn,
                "HYPERPOD_ROLE_ARN": hyperpod_role_arn,
                "INSTANCE_TYPE": instance_type,
                "INSTANCE_COUNT": str(instance_count),
                "REGION": region,
                "VF_ENV_ID": vf_env_id,
                "NOVA_MODEL": nova_model,
                "REWARD_CLUSTER_ARN": reward_cluster_arn,
                "REWARD_CPU": reward_cpu,
                "REWARD_MEMORY": reward_memory,
                "SUBNET_IDS": subnet_ids,
                "SECURITY_GROUP_ID": sg_id,
                "CUSTOM_ENV_S3_URI": custom_env_s3_uri,
                "CUSTOM_ENV_ID": custom_env_id,
                "EKS_CLUSTER_NAME": eks_cluster_name,
                "TRAINING_METHOD": training_method,
                "MAX_STEPS": max_steps,
                "GENERATION_REPLICAS": generation_replicas,
                "GLOBAL_BATCH_SIZE": global_batch_size,
                "MAX_NEW_TOKENS": max_new_tokens,
                "MAX_LENGTH": max_length,
                "TRAINING_TIMEOUT": training_timeout,
                "MLFLOW_TRACKING_ARN": mlflow_tracking_arn,
            },
            code=_lambda.Code.from_asset("lambdas/s3_trigger"),
        )

        # Suppress AwsSolutions-L1 if Python 3.13 is not yet GA in the region
        # CDK Nag checks against the latest available runtime, but Python 3.13
        # may not be available in all regions yet. This suppression can be removed
        # once Python 3.13 is generally available.
        NagSuppressions.add_resource_suppressions(
            trigger_fn,
            [
                {
                    "id": "AwsSolutions-L1",
                    "reason": "Python 3.13 is the latest available runtime in CDK. If this still triggers, it may not be GA in all regions yet.",
                }
            ],
        )

        # Wire the state machine ARN into the trigger Lambda's environment
        trigger_fn.add_environment(
            "STATE_MACHINE_ARN", self.state_machine.state_machine_arn
        )

        # S3 trigger Lambda error and throttle alarms
        cloudwatch.Alarm(
            self,
            "S3TriggerErrorAlarm",
            metric=trigger_fn.metric_errors(period=Duration.minutes(5), statistic="Sum"),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            alarm_description="S3 trigger Lambda error alarm",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        ).add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        cloudwatch.Alarm(
            self,
            "S3TriggerThrottleAlarm",
            metric=trigger_fn.metric_throttles(period=Duration.minutes(5), statistic="Sum"),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            alarm_description="S3 trigger Lambda throttle alarm",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        ).add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # Dead-letter queue for the EventBridge → Lambda target.
        #
        # Context: EventBridge already retries Lambda invocations up to 185
        # times over 24 hours with exponential backoff and jitter. For a thin
        # trigger Lambda that just calls sfn.start_execution(), exhausting all
        # retries on a transient error is extremely unlikely. The DLQ catches
        # the edge case where all retries fail (persistent misconfiguration,
        # IAM drift, etc.) so the event isn't silently lost. It also provides
        # an audit trail — operators can inspect the SQS message to see which
        # S3 upload failed to trigger the pipeline and manually re-trigger.
        #
        # Note: This is a terminal DLQ (AwsSolutions-SQS3) — messages that land
        # here require manual intervention. Adding another DLQ would create an
        # infinite chain without adding value.
        trigger_dlq = sqs.Queue(
            self,
            "TriggerDLQ",
            queue_name="nova-rl-trigger-dlq",
            retention_period=Duration.days(14),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )

        # Enforce SSL/TLS for all queue operations (AwsSolutions-SQS4)
        trigger_dlq.add_to_resource_policy(
            iam.PolicyStatement(
                sid="EnforceSSLOnly",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["sqs:*"],
                resources=[trigger_dlq.queue_arn],
                conditions={
                    "Bool": {"aws:SecureTransport": "false"}
                },
            )
        )

        # Suppress AwsSolutions-SQS3 for this terminal DLQ
        # This queue is itself a dead-letter queue (terminal queue) for EventBridge
        # failures. Adding another DLQ would create an infinite chain without value.
        # Messages that land here require manual operator intervention.
        NagSuppressions.add_resource_suppressions(
            trigger_dlq,
            [
                {
                    "id": "AwsSolutions-SQS3",
                    "reason": "This is a terminal DLQ for EventBridge trigger failures. Adding another DLQ would create an infinite chain. Messages here require manual intervention.",
                }
            ],
        )

        cdk.CfnOutput(
            scope,
            "TriggerDlqUrl",
            value=trigger_dlq.queue_url,
            description="DLQ for EventBridge trigger failures — check here if S3 uploads don't start the pipeline",
        )

        # EventBridge rule: S3 PutObject on training-data/*.jsonl
        events.Rule(
            self,
            "TrainingDataUploadRule",
            description="Triggers training pipeline when JSONL data is uploaded to training-data/",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [bucket_name]},
                    "object": {"key": [{"prefix": "training-data/"}]},
                },
            ),
            targets=[
                targets_eb.LambdaFunction(
                    trigger_fn,
                    dead_letter_queue=trigger_dlq,
                    retry_attempts=185,
                    max_event_age=Duration.hours(24),
                ),
            ],
        )
