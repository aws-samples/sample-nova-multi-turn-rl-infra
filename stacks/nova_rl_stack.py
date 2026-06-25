"""Nova Multi-Turn RL CDK Stack.

Provisions foundational AWS infrastructure for Nova model customization:
- VPC with private subnets, NAT gateway, and security group
- IAM roles for HyperPod execution and RFT infrastructure setup
- S3 bucket for training data, checkpoints, and model outputs
- SageMaker HyperPod cluster for distributed training
- Step Functions state machine for production-grade SDK orchestration

Critical scope boundary: This stack creates infrastructure only.
The Nova Forge SDK's rft_infra.setup() creates Lambda, SQS,
and DynamoDB at runtime via its own CloudFormation stack — triggered
by the Step Functions pipeline, not by CDK.
"""
import os
import tarfile

import aws_cdk as cdk
from aws_cdk import (
    aws_codebuild as codebuild,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_eks as eks,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
    aws_sagemaker as sagemaker,
)
from aws_cdk.lambda_layer_kubectl_v32 import KubectlV32Layer
from cdk_nag import NagSuppressions
from constructs import Construct
from stacks.orchestration import TrainingOrchestration
from stacks.iam_policies import create_ecs_management_policy





class NovaRlStack(cdk.Stack):
    """Single CDK stack for all Nova Multi-Turn RL infrastructure."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Configuration from CDK context ---
        instance_type = self.node.try_get_context("instance_type") or "ml.c5.xlarge"
        instance_count = int(self.node.try_get_context("instance_count") or 2)
        vpc_cidr = self.node.try_get_context("vpc_cidr") or "10.0.0.0/16"
        project_tag = self.node.try_get_context("project_tag") or "nova-multi-turn-rl"
        environment_tag = self.node.try_get_context("environment_tag") or "development"

        # Apply tags to all resources (NFR-013)
        cdk.Tags.of(self).add("project", project_tag)
        cdk.Tags.of(self).add("environment", environment_tag)
        cdk.Tags.of(self).add("managed-by", "cdk")

        # ============================================================
        # Task 3: Network and Security Foundation
        # ============================================================

        # --- VPC with private subnets and NAT gateway (FR-004, NFR-008) ---
        # ADR-004: New VPC for self-contained POC deployment
        vpc = ec2.Vpc(
            self,
            "TrainingVpc",
            ip_addresses=ec2.IpAddresses.cidr(vpc_cidr),
            max_azs=4,  # Must include use1-az6 (us-east-1d) where RIG P5 capacity is allocated
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
            ],
        )

        # VPC Flow Logs for network monitoring and security analysis (AwsSolutions-VPC7)
        vpc_flow_log_group = logs.LogGroup(
            self,
            "VpcFlowLogGroup",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        
        vpc_flow_log_role = iam.Role(
            self,
            "VpcFlowLogRole",
            assumed_by=iam.ServicePrincipal("vpc-flow-logs.amazonaws.com"),
        )
        
        ec2.FlowLog(
            self,
            "VpcFlowLog",
            resource_type=ec2.FlowLogResourceType.from_vpc(vpc),
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(
                vpc_flow_log_group,
                vpc_flow_log_role
            ),
            traffic_type=ec2.FlowLogTrafficType.ALL,
        )

        # S3 gateway endpoint for free data access (ADR-004)
        vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )

        # Interface endpoints — keeps traffic on AWS backbone, avoids NAT
        # data transfer charges, and satisfies security requirements for
        # production VPCs that may remove NAT gateways entirely.
        endpoint_sg = ec2.SecurityGroup(
            self,
            "VpcEndpointSg",
            vpc=vpc,
            description="Security group for VPC interface endpoints",
            allow_all_outbound=False,
        )
        endpoint_sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc_cidr),
            connection=ec2.Port.tcp(443),
            description="HTTPS from VPC CIDR",
        )

        for svc_name, svc in [
            ("EcrApi", ec2.InterfaceVpcEndpointAwsService.ECR),
            ("EcrDkr", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
            ("Eks", ec2.InterfaceVpcEndpointAwsService("eks")),
            ("Sts", ec2.InterfaceVpcEndpointAwsService.STS),
            ("Logs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
            # Lambda + SQS endpoints required for multi-turn RL reward
            # service communication from inside the RIG internet-free VPC.
            ("Lambda", ec2.InterfaceVpcEndpointAwsService.LAMBDA_),
            ("Sqs", ec2.InterfaceVpcEndpointAwsService.SQS),
        ]:
            vpc.add_interface_endpoint(
                f"{svc_name}Endpoint",
                service=svc,
                private_dns_enabled=True,
                security_groups=[endpoint_sg],
            )

        # Security group for HyperPod inter-node communication (FR-004)
        # Restricts traffic to only within the security group for distributed training
        training_sg = ec2.SecurityGroup(
            self,
            "TrainingSg",
            vpc=vpc,
            description="Security group for HyperPod cluster nodes - allows inter-node communication for distributed training with EFA",
            allow_all_outbound=True,
        )
        # Allow all traffic within the security group for EFA/distributed training
        # This is required for HyperPod nodes to communicate with each other
        # and does not expose the cluster to the internet (AwsSolutions-EC23)
        training_sg.add_ingress_rule(
            peer=training_sg,
            connection=ec2.Port.all_traffic(),
            description="Inter-node communication for distributed training with EFA support",
        )

        # --- HyperPod Execution IAM Role (FR-003, NFR-006) ---
        # ADR-003: Separate role for HyperPod with SageMaker/S3/CloudWatch/Bedrock
        hyperpod_role = iam.Role(
            self,
            "HyperPodExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for HyperPod cluster training workloads",
            managed_policies=[
                # Required by AWS docs for HyperPod cluster instance groups.
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerClusterInstanceRolePolicy"
                ),
            ],
        )

        # Required for training pods to invoke SDK-created Lambda functions
        # (rollout proxy). AWSLambdaRole managed policy is unavailable in
        # this account, so we grant lambda:InvokeFunction inline instead.
        hyperpod_role.add_to_policy(
            iam.PolicyStatement(
                sid="LambdaInvoke",
                actions=["lambda:InvokeFunction"],
                resources=[f"arn:aws:lambda:{self.region}:{self.account}:function:*"],
            )
        )
        
        # Suppress AwsSolutions-IAM4 for AWS managed policy required by HyperPod
        # This policy is explicitly required by AWS documentation for HyperPod cluster instance groups
        NagSuppressions.add_resource_suppressions(
            hyperpod_role,
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "AmazonSageMakerClusterInstanceRolePolicy is required by AWS for HyperPod cluster instance groups per official documentation",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSageMakerClusterInstanceRolePolicy",
                    ],
                }
            ],
        )

        # --- Customer Managed Policies for HyperPod Role ---
        
        # SageMaker permissions for training operations
        sagemaker_training_policy = iam.ManagedPolicy(
            self,
            "HyperPodSageMakerTrainingPolicy",
            description="SageMaker training operations for HyperPod cluster",
            statements=[
                iam.PolicyStatement(
                    sid="SageMakerTraining",
                    actions=[
                        "sagemaker:CreateTrainingJob",
                        "sagemaker:DescribeTrainingJob",
                        "sagemaker:StopTrainingJob",
                        "sagemaker:ListTrainingJobs",
                        "sagemaker:DescribeCluster",
                        "sagemaker:DescribeClusterNode",
                        "sagemaker:ListClusterNodes",
                    ],
                    resources=["*"],
                )
            ],
        )
        hyperpod_role.add_managed_policy(sagemaker_training_policy)
        
        # Suppress AwsSolutions-IAM5 for SageMaker wildcard
        # SageMaker training operations require wildcard for dynamic resource discovery
        NagSuppressions.add_resource_suppressions(
            sagemaker_training_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SageMaker training operations require wildcard permissions for dynamic resource discovery across training jobs and clusters",
                    "appliesTo": ["Resource::*"],
                }
            ],
        )

        # CloudWatch logging for training job observability (FR-010)
        # Broader than the managed policy (which only covers /aws/sagemaker/Clusters/*)
        cloudwatch_logs_policy = iam.ManagedPolicy(
            self,
            "HyperPodCloudWatchLogsPolicy",
            description="CloudWatch Logs for SageMaker training jobs",
            statements=[
                iam.PolicyStatement(
                    sid="CloudWatchLogs",
                    actions=[
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogStreams",
                    ],
                    resources=[
                        f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*",
                    ],
                )
            ],
        )
        hyperpod_role.add_managed_policy(cloudwatch_logs_policy)
        
        # Suppress AwsSolutions-IAM5 for CloudWatch Logs wildcard in SageMaker log groups
        # SageMaker training jobs create log groups dynamically with unpredictable names under /aws/sagemaker/
        NagSuppressions.add_resource_suppressions(
            cloudwatch_logs_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CloudWatch Logs wildcard required for SageMaker training jobs to create log groups dynamically with unpredictable names",
                    "appliesTo": [
                        "Resource::arn:aws:logs:us-east-1:<AWS::AccountId>:log-group:/aws/sagemaker/*",
                    ],
                },
            ],
        )

        # Bedrock access for Nova model artifacts
        bedrock_access_policy = iam.ManagedPolicy(
            self,
            "HyperPodBedrockAccessPolicy",
            description="Bedrock access for Nova model artifacts",
            statements=[
                iam.PolicyStatement(
                    sid="BedrockModelAccess",
                    actions=[
                        "bedrock:InvokeModel",
                        "bedrock:GetFoundationModel",
                        "bedrock:ListFoundationModels",
                        "bedrock:GetCustomModel",
                        "bedrock:CreateModelCustomizationJob",
                        "bedrock:GetModelCustomizationJob",
                    ],
                    resources=["*"],
                )
            ],
        )
        hyperpod_role.add_managed_policy(bedrock_access_policy)
        
        # Suppress AwsSolutions-IAM5 for Bedrock wildcard
        # Bedrock model operations require wildcard for model discovery and access
        NagSuppressions.add_resource_suppressions(
            bedrock_access_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Bedrock model operations require wildcard permissions for model discovery and access across foundation models",
                    "appliesTo": ["Resource::*"],
                }
            ],
        )

        # EC2 permissions for EKS-orchestrated HyperPod with custom VPC.
        # The EKS variant requires additional permissions beyond Slurm for
        # VPC CNI pod networking (assign/unassign IPs, modify ENIs) and
        # instance metadata (describe instances/types/tags).
        # See: docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-prerequisites-iam.html#sagemaker-hyperpod-prerequisites-iam-role-for-hyperpod
        ec2_networking_policy = iam.ManagedPolicy(
            self,
            "HyperPodEC2NetworkingPolicy",
            description="EC2 networking permissions for EKS-orchestrated HyperPod",
            statements=[
                iam.PolicyStatement(
                    sid="EC2NetworkingEksHyperPod",
                    actions=[
                        "ec2:AssignPrivateIpAddresses",
                        "ec2:AttachNetworkInterface",
                        "ec2:CreateNetworkInterface",
                        "ec2:CreateNetworkInterfacePermission",
                        "ec2:DeleteNetworkInterface",
                        "ec2:DeleteNetworkInterfacePermission",
                        "ec2:DescribeInstances",
                        "ec2:DescribeInstanceTypes",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DescribeTags",
                        "ec2:DescribeVpcs",
                        "ec2:DescribeDhcpOptions",
                        "ec2:DescribeSubnets",
                        "ec2:DescribeSecurityGroups",
                        "ec2:DetachNetworkInterface",
                        "ec2:ModifyNetworkInterfaceAttribute",
                        "ec2:UnassignPrivateIpAddresses",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    sid="EC2TagNetworkInterfaces",
                    actions=["ec2:CreateTags"],
                    resources=[
                        f"arn:aws:ec2:{self.region}:{self.account}:network-interface/*",
                    ],
                ),
            ],
        )
        hyperpod_role.add_managed_policy(ec2_networking_policy)

        # Suppress AwsSolutions-IAM5 for EC2 networking wildcards
        # VPC CNI pod networking requires broad EC2 permissions for dynamic ENI management
        # and ec2:CreateTags is scoped to network-interface/* (cannot be narrower)
        NagSuppressions.add_resource_suppressions(
            ec2_networking_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "EC2 networking for EKS VPC CNI requires wildcard for dynamic ENI management (AssignPrivateIpAddresses, CreateNetworkInterface, etc.). These actions operate on dynamically created resources.",
                    "appliesTo": [
                        "Resource::*",
                        f"Resource::arn:aws:ec2:us-east-1:<AWS::AccountId>:network-interface/*",
                    ],
                }
            ],
        )

        # ECR permissions for pulling container images on HyperPod nodes
        # (required for EKS-orchestrated clusters — nodes pull device plugin
        # and health monitoring agent images from ECR).
        ecr_pull_policy = iam.ManagedPolicy(
            self,
            "HyperPodEcrPullPolicy",
            description="ECR pull permissions for HyperPod nodes",
            statements=[
                iam.PolicyStatement(
                    sid="EcrPullForHyperPodNodes",
                    actions=[
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:BatchGetImage",
                        "ecr:GetAuthorizationToken",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                    resources=["*"],
                )
            ],
        )
        hyperpod_role.add_managed_policy(ecr_pull_policy)
        
        # Suppress AwsSolutions-IAM5 for ECR GetAuthorizationToken wildcard
        # GetAuthorizationToken is a global ECR action that doesn't support resource-level permissions (AWS service limitation)
        NagSuppressions.add_resource_suppressions(
            ecr_pull_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ECR GetAuthorizationToken is a global action that doesn't support resource-level permissions per AWS service design",
                    "appliesTo": ["Resource::*"],
                },
            ],
        )

        # SQS permissions for HyperPod pods (vLLM generation pods poll
        # GenerateRequestQueue and write to GenerateResponseQueue)
        hyperpod_sqs_policy = iam.ManagedPolicy(
            self,
            "HyperPodSqsPolicy",
            description="SQS access for HyperPod training/generation pods",
            statements=[
                iam.PolicyStatement(
                    sid="SqsAccess",
                    actions=[
                        "sqs:SendMessage",
                        "sqs:ReceiveMessage",
                        "sqs:DeleteMessage",
                        "sqs:GetQueueAttributes",
                        "sqs:GetQueueUrl",
                    ],
                    resources=[
                        f"arn:aws:sqs:{self.region}:{self.account}:{self.node.try_get_context('sdk_resource_prefix') or 'nrl'}-*-NovaForgeSDK-SageMaker-*.fifo",
                    ],
                )
            ],
        )
        hyperpod_role.add_managed_policy(hyperpod_sqs_policy)

        # Suppress AwsSolutions-IAM5 for SQS wildcard — SDK creates queues with dynamic date-based names
        NagSuppressions.add_resource_suppressions(
            hyperpod_sqs_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK creates SQS FIFO queues with dynamic names containing dates; resource pattern scoped to sdk_resource_prefix",
                    "appliesTo": [f"Resource::arn:aws:sqs:{self.region}:{self.account}:{self.node.try_get_context('sdk_resource_prefix') or 'nrl'}-*-NovaForgeSDK-SageMaker-*.fifo"],
                },
            ],
        )

        # --- RFT Execution IAM Role (FR-003, NFR-006) ---
        # ADR-003: Separate role for SDK infrastructure setup
        rft_role = iam.Role(
            self,
            "RftExecutionRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            ),
            description="Execution role for RFT SDK infrastructure setup",
        )

        # --- Customer Managed Policies for RFT Role ---
        
        # CloudFormation permissions for SDK stack creation (FR-006)
        # SAM CLI requires additional permissions beyond basic stack operations
        cloudformation_management_policy = iam.ManagedPolicy(
            self,
            "RftCloudFormationManagementPolicy",
            description="CloudFormation stack management for SDK infrastructure",
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
                        f"arn:aws:cloudformation:{self.region}:{self.account}:stack/*",
                        f"arn:aws:cloudformation:{self.region}:{self.account}:stackset/*",
                    ],
                )
            ],
        )
        rft_role.add_managed_policy(cloudformation_management_policy)

        # Suppress AwsSolutions-IAM5 for CloudFormation management wildcards
        # SDK creates CloudFormation stacks with dynamic names; stack/* and stackset/* are the tightest patterns
        NagSuppressions.add_resource_suppressions(
            cloudformation_management_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK creates CloudFormation stacks with dynamic names. stack/* and stackset/* are the tightest patterns possible for SDK-managed infrastructure.",
                    "appliesTo": [
                        f"Resource::arn:aws:cloudformation:us-east-1:<AWS::AccountId>:stack/*",
                        f"Resource::arn:aws:cloudformation:us-east-1:<AWS::AccountId>:stackset/*",
                    ],
                }
            ],
        )

        # Lambda permissions for SDK-created functions
        lambda_management_policy = iam.ManagedPolicy(
            self,
            "RftLambdaManagementPolicy",
            description="Lambda function management for SDK infrastructure",
            statements=[
                iam.PolicyStatement(
                    sid="LambdaManagement",
                    actions=[
                        "lambda:CreateFunction",
                        "lambda:DeleteFunction",
                        "lambda:UpdateFunctionCode",
                        "lambda:UpdateFunctionConfiguration",
                        "lambda:GetFunction",
                        "lambda:InvokeFunction",
                        "lambda:AddPermission",
                        "lambda:RemovePermission",
                        "lambda:CreateFunctionUrlConfig",
                        "lambda:GetFunctionUrlConfig",
                        "lambda:DeleteFunctionUrlConfig",
                    ],
                    resources=[
                        f"arn:aws:lambda:{self.region}:{self.account}:function:*",
                    ],
                )
            ],
        )
        rft_role.add_managed_policy(lambda_management_policy)

        # Suppress AwsSolutions-IAM5 for Lambda management wildcard
        # SDK creates Lambda functions with dynamic names; function:* is required
        NagSuppressions.add_resource_suppressions(
            lambda_management_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK creates Lambda functions with dynamic names. function:* wildcard is required for SDK-managed Lambda lifecycle operations.",
                    "appliesTo": [
                        f"Resource::arn:aws:lambda:us-east-1:<AWS::AccountId>:function:*",
                    ],
                }
            ],
        )

        # SQS permissions for SDK-created FIFO queues
        sqs_management_policy = iam.ManagedPolicy(
            self,
            "RftSqsManagementPolicy",
            description="SQS queue management for SDK infrastructure",
            statements=[
                iam.PolicyStatement(
                    sid="SqsManagement",
                    actions=[
                        "sqs:CreateQueue",
                        "sqs:DeleteQueue",
                        "sqs:GetQueueAttributes",
                        "sqs:SetQueueAttributes",
                        "sqs:SendMessage",
                        "sqs:ReceiveMessage",
                        "sqs:DeleteMessage",
                        "sqs:GetQueueUrl",
                        "sqs:TagQueue",
                    ],
                    resources=[
                        f"arn:aws:sqs:{self.region}:{self.account}:*",
                    ],
                )
            ],
        )
        rft_role.add_managed_policy(sqs_management_policy)

        # Suppress AwsSolutions-IAM5 for SQS management wildcard
        # SDK creates SQS queues with dynamic names
        NagSuppressions.add_resource_suppressions(
            sqs_management_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK creates SQS FIFO queues with dynamic names. Wildcard is required for SDK-managed queue lifecycle operations.",
                    "appliesTo": [
                        f"Resource::arn:aws:sqs:us-east-1:<AWS::AccountId>:*",
                    ],
                }
            ],
        )

        # DynamoDB permissions for SDK-created table
        dynamodb_management_policy = iam.ManagedPolicy(
            self,
            "RftDynamoDbManagementPolicy",
            description="DynamoDB table management for SDK infrastructure",
            statements=[
                iam.PolicyStatement(
                    sid="DynamoDbManagement",
                    actions=[
                        "dynamodb:CreateTable",
                        "dynamodb:DeleteTable",
                        "dynamodb:DescribeTable",
                        "dynamodb:PutItem",
                        "dynamodb:GetItem",
                        "dynamodb:UpdateItem",
                        "dynamodb:Query",
                        "dynamodb:Scan",
                        "dynamodb:TagResource",
                    ],
                    resources=[
                        f"arn:aws:dynamodb:{self.region}:{self.account}:table/*",
                    ],
                )
            ],
        )
        rft_role.add_managed_policy(dynamodb_management_policy)

        # Suppress AwsSolutions-IAM5 for DynamoDB management wildcard
        # SDK creates DynamoDB tables with dynamic names
        NagSuppressions.add_resource_suppressions(
            dynamodb_management_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK creates DynamoDB tables with dynamic names. table/* wildcard is required for SDK-managed table lifecycle operations.",
                    "appliesTo": [
                        f"Resource::arn:aws:dynamodb:us-east-1:<AWS::AccountId>:table/*",
                    ],
                }
            ],
        )

        # ECS permissions for reward worker deployment (FR-007)
        ecs_management_policy = create_ecs_management_policy(
            self, "RftEcsManagementPolicy",
        )
        rft_role.add_managed_policy(ecs_management_policy)

        # IAM PassRole for SDK to assign roles to Lambda/ECS
        # Note: Only includes hyperpod_role here. The rft_role self-reference
        # is handled in orchestration.py to avoid circular dependencies.
        iam_passrole_policy = iam.ManagedPolicy(
            self,
            "RftIamPassRolePolicy",
            description="IAM PassRole for SDK to assign roles to Lambda/ECS",
            statements=[
                iam.PolicyStatement(
                    sid="IamPassRole",
                    actions=["iam:PassRole"],
                    resources=[
                        hyperpod_role.role_arn,
                    ],
                )
            ],
        )
        rft_role.add_managed_policy(iam_passrole_policy)

        # IAM role management for SDK-created roles
        iam_role_management_policy = iam.ManagedPolicy(
            self,
            "RftIamRoleManagementPolicy",
            description="IAM role management for SDK-created roles",
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
                    ],
                    resources=[
                        f"arn:aws:iam::{self.account}:role/*nova*",
                        f"arn:aws:iam::{self.account}:role/*rft*",
                        f"arn:aws:iam::{self.account}:role/*Nova*",
                        f"arn:aws:iam::{self.account}:role/*RFT*",
                    ],
                )
            ],
        )
        rft_role.add_managed_policy(iam_role_management_policy)

        # Suppress AwsSolutions-IAM5 for IAM role management wildcards
        # SDK creates IAM roles with dynamic names containing nova/rft patterns
        NagSuppressions.add_resource_suppressions(
            iam_role_management_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SDK dynamically creates IAM roles with unpredictable names. Wildcards scoped to *nova*/*rft*/*Nova*/*RFT* patterns are the tightest constraint possible.",
                    "appliesTo": [
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*nova*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*rft*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*Nova*",
                        f"Resource::arn:aws:iam::<AWS::AccountId>:role/*RFT*",
                    ],
                }
            ],
        )

        # CloudWatch logs for SDK-created Lambda functions
        cloudwatch_logs_rft_policy = iam.ManagedPolicy(
            self,
            "RftCloudWatchLogsPolicy",
            description="CloudWatch Logs for SDK-created Lambda functions",
            statements=[
                iam.PolicyStatement(
                    sid="CloudWatchLogsRft",
                    actions=[
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogGroups",
                        "logs:DescribeLogStreams",
                        "logs:GetLogEvents",
                    ],
                    resources=[
                        f"arn:aws:logs:{self.region}:{self.account}:log-group:*",
                    ],
                )
            ],
        )
        rft_role.add_managed_policy(cloudwatch_logs_rft_policy)
        
        # Suppress AwsSolutions-IAM5 for CloudWatch Logs wildcard for SDK-created Lambda functions
        # The SDK creates Lambda functions dynamically with unpredictable names, requiring wildcard log group access
        NagSuppressions.add_resource_suppressions(
            cloudwatch_logs_rft_policy,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CloudWatch Logs wildcard required for SDK-created Lambda functions to create log groups dynamically with unpredictable names",
                    "appliesTo": [
                        "Resource::arn:aws:logs:us-east-1:<AWS::AccountId>:log-group:*",
                    ],
                },
            ],
        )

        # ============================================================
        # Task 4: Storage and Compute
        # ============================================================

        # --- S3 Bucket (FR-002, NFR-007, NFR-009) ---
        # ADR-002: Single bucket with prefix-based organization
        nova_model = self.node.try_get_context("nova_model") or "NOVA_LITE_2"
        vf_env_id = self.node.try_get_context("vf_env_id") or "wordle"
        reward_cpu = self.node.try_get_context("reward_cpu") or "2048"
        reward_memory = self.node.try_get_context("reward_memory") or "4096"
        retain_bucket = (self.node.try_get_context("retain_bucket") or "true").lower() == "true"
        removal_policy = cdk.RemovalPolicy.RETAIN if retain_bucket else cdk.RemovalPolicy.DESTROY

        # Custom environment support: when enabled, build a tarball at synth
        # time so BucketDeployment uploads it to S3 during cdk deploy.
        use_custom_env = (self.node.try_get_context("use_custom_env") or "false").lower() == "true"
        custom_env_id = self.node.try_get_context("custom_env_id") or "my-custom-env"
        custom_env_s3_uri = ""
        # Custom env and vf_env_id are mutually exclusive in the SDK
        if use_custom_env:
            vf_env_id = ""

        training_bucket = s3.Bucket(
            self,
            "TrainingBucket",
            encryption=s3.BucketEncryption.KMS_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            event_bridge_enabled=True,
            removal_policy=removal_policy,
            auto_delete_objects=not retain_bucket,
        )

        # Suppress AwsSolutions-S1 for S3 access logging
        # POC scope: access logging adds cost without compliance requirement
        NagSuppressions.add_resource_suppressions(
            training_bucket,
            [
                {
                    "id": "AwsSolutions-S1",
                    "reason": "POC deployment: S3 access logging not required. Bucket has encryption, SSL enforcement, versioning, and public access blocking enabled.",
                }
            ],
        )

        # Grant S3 access to both roles
        training_bucket.grant_read_write(hyperpod_role)
        training_bucket.grant_read_write(rft_role)
        
        # Suppress AwsSolutions-IAM5 for S3 grant_read_write wildcards
        # S3 object operations require /* wildcard for bucket access (CDK-generated policy)
        NagSuppressions.add_resource_suppressions(
            hyperpod_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 object operations require /* wildcard for bucket access - this is a CDK-generated policy from grant_read_write",
                    "appliesTo": [
                        "Action::s3:GetObject*",
                        "Action::s3:GetBucket*",
                        "Action::s3:List*",
                        "Action::s3:DeleteObject*",
                        "Action::s3:PutObject",
                        "Action::s3:PutObjectLegalHold",
                        "Action::s3:PutObjectRetention",
                        "Action::s3:PutObjectTagging",
                        "Action::s3:PutObjectVersionTagging",
                        "Action::s3:Abort*",
                        f"Resource::{training_bucket.bucket_arn}/*",
                    ],
                }
            ],
            apply_to_children=True,
        )
        
        NagSuppressions.add_resource_suppressions(
            rft_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 object operations require /* wildcard for bucket access - this is a CDK-generated policy from grant_read_write",
                    "appliesTo": [
                        "Action::s3:GetObject*",
                        "Action::s3:GetBucket*",
                        "Action::s3:List*",
                        "Action::s3:DeleteObject*",
                        "Action::s3:PutObject",
                        "Action::s3:PutObjectLegalHold",
                        "Action::s3:PutObjectRetention",
                        "Action::s3:PutObjectTagging",
                        "Action::s3:PutObjectVersionTagging",
                        "Action::s3:Abort*",
                        f"Resource::{training_bucket.bucket_arn}/*",
                    ],
                }
            ],
            apply_to_children=True,
        )

        # Build and upload custom environment tarball when enabled.
        # The SDK's CommonInfraCommands downloads this tarball into the
        # Fargate container and pip-installs it as the reward environment.
        if use_custom_env:
            env_source = os.path.join("custom-environments", custom_env_id)
            if not os.path.isdir(env_source):
                raise ValueError(
                    f"Custom environment directory not found: {env_source}. "
                    f"Create it or set use_custom_env to false."
                )
            tar_path = os.path.join("custom-environments", f"{custom_env_id}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(env_source, arcname=custom_env_id)

            s3_deploy.BucketDeployment(
                self,
                "CustomEnvDeployment",
                sources=[s3_deploy.Source.asset("custom-environments/", exclude=["*/"])],
                destination_bucket=training_bucket,
                destination_key_prefix="custom-envs",
            )
            custom_env_s3_uri = f"s3://{training_bucket.bucket_name}/custom-envs/{custom_env_id}.tar.gz"

        # --- HyperPod Cluster (FR-001, NFR-004, NFR-005) ---
        # ADR-001: CfnCluster L1 construct (no L2 available)
        # EKS-based orchestration: the HyperPod cluster uses an EKS cluster
        # as its orchestrator instead of Slurm. This is required because the
        # Nova Forge SDK's SMHPRuntimeManager uses the `hyperpod` CLI
        # which requires an EKS-based HyperPod cluster.
        private_subnet_ids = [
            subnet.subnet_id for subnet in vpc.private_subnets
        ]

        # Upload lifecycle scripts to S3 before cluster creation
        lifecycle_deploy = s3_deploy.BucketDeployment(
            self,
            "LifecycleScripts",
            sources=[s3_deploy.Source.asset("lifecycle-scripts")],
            destination_bucket=training_bucket,
            destination_key_prefix="lifecycle-scripts",
        )

        # Upload CodeBuild submit script for Step 4
        s3_deploy.BucketDeployment(
            self,
            "CodeBuildScripts",
            sources=[s3_deploy.Source.asset("lambdas/submit_training", exclude=["__pycache__"])],
            destination_bucket=training_bucket,
            destination_key_prefix="codebuild-scripts",
        )

        # --- EKS Cluster for HyperPod orchestration ---
        # The HyperPod cluster requires an EKS cluster as its orchestrator.
        # CDK's eks.Cluster with kubectl_enabled creates a kubectl Lambda
        # layer that allows Helm chart installation as a CDK resource.
        k8s_version = self.node.try_get_context("eks_kubernetes_version") or "1.32"
        eks_admin_role_arn = self.node.try_get_context("eks_admin_role_arn") or ""

        eks_secrets_key = kms.Key(self, "EksSecretsKey",
            description="KMS key for EKS Kubernetes secrets envelope encryption",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # EKS endpoint: HyperPod requires PUBLIC. The /readyz health check runs
        # from the KAS Proxy Agent in AWS service accounts outside the customer
        # VPC. PUBLIC_AND_PRIVATE fails (18+ deploys, P394505130, V2160654831).
        # Service CIDR ranges are unpublished, so public_access_cidrs can't be
        # restricted. Access protected by IAM authentication.

        eks_cluster = eks.Cluster(
            self,
            "HyperPodEksCluster",
            cluster_name=f"{project_tag}-eks",
            version=eks.KubernetesVersion.of(k8s_version),
            vpc=vpc,
            vpc_subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)],
            default_capacity=0,  # HyperPod manages its own nodes
            endpoint_access=eks.EndpointAccess.PUBLIC,
            kubectl_layer=KubectlV32Layer(self, "KubectlLayer"),
            authentication_mode=eks.AuthenticationMode.API_AND_CONFIG_MAP,
            secrets_encryption_key=eks_secrets_key,
        )

        # System node group — Helm chart pods (training-operators, mpi-operator,
        # coredns, etc.) need nodes to schedule on BEFORE HyperPod creates its
        # RIG nodes. Without this, it's a chicken-and-egg: HyperPod validates
        # Helm pods are running, but pods can't run without nodes.
        eks_cluster.add_nodegroup_capacity("SystemNodes",
            instance_types=[ec2.InstanceType("t3.medium")],
            min_size=1,
            max_size=2,
            desired_size=1,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        )
        NagSuppressions.add_resource_suppressions(
            eks_cluster.node.find_child("NodegroupSystemNodes"),
            [{"id": "AwsSolutions-IAM4", "reason": "EKS managed node group requires AWS managed policies (AmazonEKSWorkerNodePolicy, AmazonEKS_CNI_Policy, AmazonEC2ContainerRegistryReadOnly)."}],
            apply_to_children=True,
        )

        # Core EKS addons — CDK's Custom Resource doesn't auto-install these
        eks_addons = []
        for addon_name in ["vpc-cni", "coredns", "kube-proxy", "eks-pod-identity-agent"]:
            addon = eks.CfnAddon(self, f"EksAddon-{addon_name}",
                addon_name=addon_name,
                cluster_name=eks_cluster.cluster_name,
            )
            eks_addons.append(addon)

        # Suppress AwsSolutions-EKS1 for public API endpoint
        # POC requires public+private endpoint access for developer kubectl access
        # and HyperPod CLI operations from outside the VPC
        NagSuppressions.add_resource_suppressions(
            eks_cluster,
            [
                {
                    "id": "AwsSolutions-EKS1",
                    "reason": "SageMaker HyperPod requires PUBLIC EKS endpoint. The /readyz health check runs from the KAS Proxy Agent in AWS service accounts outside the customer VPC — PUBLIC_AND_PRIVATE fails (18+ deploys, P394505130, V2160654831). Service CIDR ranges are unpublished so public_access_cidrs can't be restricted. Access protected by IAM authentication.",
                },
                {
                    "id": "AwsSolutions-EKS2",
                    "reason": "POC deployment: EKS control plane logging (api, audit, authenticator, controllerManager, scheduler) adds cost without compliance requirement for POC scope.",
                },
            ],
            apply_to_children=True,
        )

        # Grant the HyperPod execution role access to the EKS cluster so
        # SageMaker can register HyperPod nodes as EKS worker nodes.
        eks_cluster.aws_auth.add_role_mapping(
            hyperpod_role,
            groups=["system:masters"],
            username="sagemaker-hyperpod",
        )

        # NOTE: The HyperPod service-linked role (AWSServiceRoleForSageMakerHyperPod)
        # EKS access entries are created automatically by SageMaker during
        # HyperPod cluster creation. The EKS API does not allow external
        # callers to create access entries for SLRs — only the owning service can.

        # If an admin role ARN is provided, grant it cluster access too
        # (useful for debugging with kubectl from a dev machine).
        if eks_admin_role_arn:
            admin_role = iam.Role.from_role_arn(
                self, "EksAdminRole", eks_admin_role_arn
            )
            eks_cluster.aws_auth.add_masters_role(admin_role)

        # VPC CNI: EKS 1.31 ships with VPC CNI >= 1.18.3 by default,
        # which meets the HyperPod prerequisite. No explicit install needed.

        # --- HyperPod Helm Chart Installation ---
        # Installs HyperPod dependencies (Training Operator, MPI operator,
        # health monitoring, EFA/NVIDIA device plugins) via CodeBuild.
        # The CodeBuild job clones the RIG-specific revision from
        # github.com/aws/sagemaker-hyperpod-cli and runs helm install
        # + install_rig_dependencies.sh directly on the EKS cluster.

        # KMS key for Helm install CodeBuild project encryption (AwsSolutions-CB4)
        helm_codebuild_kms_key = kms.Key(
            self,
            "HelmCodeBuildEncryptionKey",
            description="KMS key for encrypting Helm install CodeBuild project artifacts and logs",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # CodeBuild project that installs the Helm chart from S3
        helm_install_project = codebuild.Project(
            self,
            "HyperPodHelmInstallProject",
            project_name=f"{project_tag}-helm-install",
            description="Installs pre-resolved HyperPod Helm chart on EKS cluster",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.SMALL,
            ),
            encryption_key=helm_codebuild_kms_key,
            timeout=cdk.Duration.minutes(30),
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "commands": [
                            "curl -LO https://dl.k8s.io/release/v1.32.0/bin/linux/amd64/kubectl",
                            "chmod +x kubectl && mv kubectl /usr/local/bin/kubectl",
                            "curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3",
                            "chmod 700 get_helm.sh && ./get_helm.sh && rm -f get_helm.sh",
                            # yq v4 required by install_rig_dependencies.sh
                            "curl -fsSL -o /usr/local/bin/yq https://github.com/mikefarah/yq/releases/download/v4.44.1/yq_linux_amd64",
                            "chmod +x /usr/local/bin/yq",
                        ],
                    },
                    "pre_build": {
                        "commands": [
                            f"aws eks update-kubeconfig --name {project_tag}-eks --region $AWS_DEFAULT_REGION",
                            'EXISTING=$(helm list -n kube-system -q | grep "^hyperpod-dependencies$" || true)',
                            'if [ -n "$EXISTING" ]; then echo "HyperPod Helm chart already installed. Skipping."; exit 0; fi',
                            # Clone the RIG-specific Helm chart revision
                            "git clone https://github.com/aws/sagemaker-hyperpod-cli.git /tmp/helm-repo",
                            "cd /tmp/helm-repo && git checkout c5275ddbbca58164d1f5bd3a2811e0fc952f7ff4",
                        ],
                    },
                    "build": {
                        "commands": [
                            # Step 1: Install standard HyperPod Helm chart with regional values
                            "cd /tmp/helm-repo/helm_chart/HyperPodHelmChart",
                            "helm dependency update .",
                            f'REGION_VALUES=""; if [ -f "regional-values/values-$AWS_DEFAULT_REGION.yaml" ]; then REGION_VALUES="-f regional-values/values-$AWS_DEFAULT_REGION.yaml"; fi',
                            f'helm install hyperpod-dependencies . --namespace kube-system --set health-monitoring-agent.region=$AWS_DEFAULT_REGION $REGION_VALUES',
                            'echo "Standard HyperPod Helm chart installed"',
                            # Wait for Helm-installed resources before RIG script
                            "kubectl wait --for=condition=available deployment/hyperpod-dependencies-training-operators -n kubeflow --timeout=300s",
                            "kubectl wait --for=condition=available deployment/hyperpod-dependencies-mpi-operator -n kube-system --timeout=300s",
                            "kubectl rollout status daemonset/hyperpod-dependencies-aws-efa-k8s-device-plugin -n kube-system --timeout=300s",
                            # Step 2: Run RIG dependencies install script
                            # This modifies CoreDNS (converts to DaemonSet), VPC CNI
                            # (creates RIG-specific aws-node), patches training-operators,
                            # mpi-operator, and EFA for RIG node tolerations.
                            "cd /tmp/helm-repo/helm_chart",
                            'echo "y" | ./install_rig_dependencies.sh',
                            'echo "RIG dependencies installed"',
                            # Step 3: Delete kueue (not supported by RIG)
                            'kubectl delete deployment kueue-controller-manager -n kueue-system --ignore-not-found=true',
                            'kubectl delete mutatingwebhookconfiguration kueue-mutating-webhook-configuration --ignore-not-found=true',
                            'kubectl delete validatingwebhookconfiguration kueue-validating-webhook-configuration --ignore-not-found=true',
                        ],
                    },
                },
            }),
        )

        helm_install_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["eks:DescribeCluster", "eks:AccessKubernetesApi"],
                resources=[eks_cluster.cluster_arn],
            )
        )
        helm_install_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:ListBucket"],
                resources=[
                    training_bucket.bucket_arn,
                    f"{training_bucket.bucket_arn}/*",
                ],
            )
        )

        eks_cluster.aws_auth.add_role_mapping(
            helm_install_project.role,
            groups=["system:masters"],
            username="codebuild-helm-install",
        )

        # Custom Resource triggers CodeBuild at deploy time and waits for completion.
        # The Lambda polls build status every 15s until success or failure.
        hyperpod_helm = cdk.CustomResource(
            self,
            "HyperPodHelmInstallCR",
            service_token=cdk.custom_resources.Provider(
                self,
                "HelmInstallProvider",
                on_event_handler=_lambda.Function(
                    self,
                    "HelmInstallTriggerFn",
                    runtime=_lambda.Runtime.PYTHON_3_12,
                    handler="handler.handler",
                    timeout=cdk.Duration.minutes(14),
                    memory_size=128,
                    code=_lambda.Code.from_asset("lambdas/helm_install_trigger"),
                    initial_policy=[
                        iam.PolicyStatement(
                            actions=["codebuild:StartBuild", "codebuild:BatchGetBuilds"],
                            resources=[helm_install_project.project_arn],
                        ),
                    ],
                ),
            ).service_token,
            properties={
                "ProjectName": helm_install_project.project_name,
                "Version": "1",
            },
        )
        hyperpod_helm.node.add_dependency(eks_cluster)
        for addon in eks_addons:
            hyperpod_helm.node.add_dependency(addon)

        # Suppress AwsSolutions-IAM5 for Helm install CodeBuild project
        # CodeBuild needs S3 access for chart download, KMS for encryption, and log groups
        NagSuppressions.add_resource_suppressions(
            helm_install_project,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodeBuild Helm install project requires wildcard permissions for S3 chart download, KMS encryption operations, and CloudWatch log group management. These are CDK-generated default policies.",
                    "appliesTo": [
                        f"Resource::<TrainingBucketEB7BB5C9.Arn>/*",
                        "Action::kms:ReEncrypt*",
                        "Action::kms:GenerateDataKey*",
                    ],
                }
            ],
            apply_to_children=True,
        )

        # Suppress AwsSolutions-IAM4 and L1 for Helm install trigger Lambda
        # CDK Custom Resource provider uses AWSLambdaBasicExecutionRole and may not use latest runtime
        NagSuppressions.add_resource_suppressions(
            hyperpod_helm,
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CDK Custom Resource framework uses AWSLambdaBasicExecutionRole managed policy. This is a CDK internal construct.",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                    ],
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "CDK Custom Resource framework Lambda runtime is managed by CDK and may not use the latest runtime version.",
                },
            ],
            apply_to_children=True,
        )

        # EKS permissions for the HyperPod execution role — needed so
        # SageMaker can describe the EKS cluster during HyperPod creation.
        eks_describe_policy = iam.ManagedPolicy(
            self,
            "HyperPodEksDescribePolicy",
            description="EKS cluster describe permissions for HyperPod",
            statements=[
                iam.PolicyStatement(
                    sid="EksDescribeForHyperPod",
                    actions=[
                        "eks:DescribeCluster",
                        "eks:ListNodegroups",
                        "eks:DescribeNodegroup",
                    ],
                    resources=[eks_cluster.cluster_arn],
                )
            ],
        )
        hyperpod_role.add_managed_policy(eks_describe_policy)

        # Required for EKS Pod Identity — HyperPod nodes use this to assume
        # the execution role via the eks-pod-identity-agent addon.
        hyperpod_role.add_to_policy(
            iam.PolicyStatement(
                sid="EksPodIdentity",
                actions=["eks-auth:AssumeRoleForPodIdentity"],
                resources=[eks_cluster.cluster_arn],
            )
        )

        # Select the last private subnet (us-east-1d / use1-az6) for RIG placement.
        # Both working clusters (riv-rig, rft-poc-2) pin OverrideVpcConfig to a
        # single subnet in use1-az6. Without this, SageMaker fails with generic
        # "Request to service failed" — likely can't find RIG capacity in other AZs.
        rig_subnet_id = private_subnet_ids[-1]

        cluster = sagemaker.CfnCluster(
            self,
            "HyperPodCluster",
            cluster_name=f"{project_tag}-cluster",
            node_recovery="Automatic",
            restricted_instance_groups=[
                sagemaker.CfnCluster.ClusterRestrictedInstanceGroupProperty(
                    execution_role=hyperpod_role.role_arn,
                    instance_count=instance_count,
                    instance_group_name="training-group",
                    instance_type=instance_type,
                    threads_per_core=1,
                    instance_storage_configs=[
                        sagemaker.CfnCluster.ClusterInstanceStorageConfigProperty(
                            ebs_volume_config=sagemaker.CfnCluster.ClusterEbsVolumeConfigProperty(
                                volume_size_in_gb=500,
                            ),
                        ),
                    ],
                    environment_config=sagemaker.CfnCluster.EnvironmentConfigProperty(
                        f_sx_lustre_config=sagemaker.CfnCluster.FSxLustreConfigProperty(
                            per_unit_storage_throughput=250,
                            size_in_gib=9600,
                        ),
                    ),
                    override_vpc_config=sagemaker.CfnCluster.VpcConfigProperty(
                        security_group_ids=[training_sg.security_group_id],
                        subnets=[rig_subnet_id],
                    ),
                ),
            ],
            orchestrator=sagemaker.CfnCluster.OrchestratorProperty(
                eks=sagemaker.CfnCluster.ClusterOrchestratorEksConfigProperty(
                    cluster_arn=eks_cluster.cluster_arn,
                ),
            ),
            vpc_config=sagemaker.CfnCluster.VpcConfigProperty(
                security_group_ids=[training_sg.security_group_id],
                subnets=private_subnet_ids,
            ),
        )

        # Remove CDK auto-tags from the CfnCluster — both working console-
        # created clusters (rft-poc-2, riv-rig) have zero Tags on the
        # AWS::SageMaker::Cluster resource.  CDK propagates stack-level
        # Tags.of(self).add() to every child; exclude this resource.
        cdk.Tags.of(cluster).remove("project")
        cdk.Tags.of(cluster).remove("environment")
        cdk.Tags.of(cluster).remove("managed-by")

        # Ensure the HyperPod role AND its inline policies are fully created
        # before the cluster. The CfnCluster depends on the role ARN (implicit)
        # but NOT on the policy resource. Without this, SageMaker assumes the
        # role before ec2:DescribeSubnets is attached, causing
        # "Unable to retrieve subnets" during cluster creation.
        cluster.node.add_dependency(hyperpod_role)
        # Also depend on the default policy node (where inline statements live)
        for child in hyperpod_role.node.children:
            cluster.node.add_dependency(child)
        # Lifecycle scripts must be in S3 before cluster creation
        cluster.node.add_dependency(lifecycle_deploy)
        # EKS cluster must be ready before HyperPod creation.
        cluster.node.add_dependency(eks_cluster)
        # HyperPod Helm chart must be installed before cluster creation.
        # Without this, the CfnCluster fails with "missing one or more
        # required dependencies" because the EKS cluster lacks the
        # Kubeflow Training Operator, health monitoring agent, and other
        # controllers that HyperPod expects.
        cluster.node.add_dependency(hyperpod_helm)
        # aws_auth ConfigMap must be applied before HyperPod creation.
        # Without this, HyperPod nodes can't authenticate to EKS because
        # the execution role isn't mapped to system:masters yet.
        cluster.node.add_dependency(eks_cluster.aws_auth)

        # --- ECS Cluster for SDK reward workers (FR-007) ---
        # The SDK's RFTMultiturnInfrastructure uses this cluster to run
        # reward environment tasks on Fargate when infrastructure_arn
        # points to an ECS cluster ARN (production mode).
        reward_cluster = ecs.Cluster(
            self,
            "RewardWorkerCluster",
            cluster_name=f"{project_tag}-reward-workers",
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        # ============================================================
        # CfnOutputs - Consumed by setup.sh (FR-005)
        # ============================================================

        # Network outputs
        cdk.CfnOutput(self, "VpcId", value=vpc.vpc_id, description="VPC ID")
        cdk.CfnOutput(
            self,
            "PrivateSubnetIds",
            value=",".join(private_subnet_ids),
            description="Private subnet IDs (comma-separated)",
        )
        cdk.CfnOutput(
            self,
            "SecurityGroupId",
            value=training_sg.security_group_id,
            description="Training security group ID",
        )

        # IAM outputs
        cdk.CfnOutput(
            self,
            "HyperPodRoleArn",
            value=hyperpod_role.role_arn,
            description="HyperPod execution role ARN",
        )
        cdk.CfnOutput(
            self,
            "RftRoleArn",
            value=rft_role.role_arn,
            description="RFT execution role ARN",
        )

        # Storage outputs
        cdk.CfnOutput(
            self,
            "TrainingBucketName",
            value=training_bucket.bucket_name,
            description="S3 bucket name for training artifacts",
        )
        cdk.CfnOutput(
            self,
            "TrainingBucketArn",
            value=training_bucket.bucket_arn,
            description="S3 bucket ARN",
        )

        # Compute outputs
        cdk.CfnOutput(
            self,
            "ClusterName",
            value=f"{project_tag}-cluster",
            description="HyperPod cluster name",
        )
        cdk.CfnOutput(
            self,
            "InstanceType",
            value=instance_type,
            description="HyperPod instance type",
        )
        cdk.CfnOutput(
            self,
            "InstanceCount",
            value=str(instance_count),
            description="HyperPod instance count",
        )
        cdk.CfnOutput(
            self,
            "RewardClusterArn",
            value=reward_cluster.cluster_arn,
            description="ECS cluster ARN for SDK reward workers",
        )
        cdk.CfnOutput(
            self,
            "EksClusterName",
            value=eks_cluster.cluster_name,
            description="EKS cluster name for HyperPod orchestration",
        )
        cdk.CfnOutput(
            self,
            "EksClusterArn",
            value=eks_cluster.cluster_arn,
            description="EKS cluster ARN",
        )

        # ============================================================
        # Stack-level NagSuppressions for CDK internal constructs
        # These constructs are created by CDK and cannot be directly referenced.
        # ============================================================
        NagSuppressions.add_stack_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CDK internal constructs (BucketDeployment, EKS ClusterResourceProvider, KubectlProvider, Custom Resource framework) use AWS managed policies. These are CDK-managed and cannot be customized.",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSClusterPolicy",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonElasticContainerRegistryPublicReadOnly",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSVPCResourceController",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSWorkerNodePolicy",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CDK internal constructs (BucketDeployment, EKS ClusterResourceProvider, KubectlProvider, Custom Resource framework) require wildcard permissions for dynamic resource management. These are CDK-managed.",
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "CDK internal Lambda functions (BucketDeployment, EKS providers, Custom Resource framework) use runtimes managed by CDK. Runtime versions are updated with CDK library upgrades.",
                },
                {
                    "id": "AwsSolutions-SF1",
                    "reason": "CDK internal Step Functions (EKS ClusterResourceProvider) logging configuration is managed by CDK.",
                },
                {
                    "id": "AwsSolutions-SF2",
                    "reason": "CDK internal Step Functions (EKS ClusterResourceProvider) X-Ray tracing configuration is managed by CDK.",
                },
                {
                    "id": "CdkNagValidationFailure",
                    "reason": "Security group rule references VPC CIDR via intrinsic function (Fn::GetAtt). cdk-nag cannot validate intrinsic function values at synth time.",
                },
            ],
        )

        # Path-based suppression for KubectlHandlerRole conditional ECR public policy
        # CDK generates an Fn::If conditional for AmazonElasticContainerRegistryPublicReadOnly
        # that cannot be matched by appliesTo patterns
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            f"/{self.stack_name}/HyperPodEksCluster/KubectlHandlerRole/Resource",
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CDK EKS KubectlHandlerRole uses AWS managed policies including conditional ECR public read policy (Fn::If). These are CDK-managed and required for EKS kubectl operations.",
                },
            ],
        )

        # Path-based suppressions for CDK-internal EKS nested constructs.
        # These live outside the normal construct tree at @aws-cdk--aws-eks.*
        # and cannot be reached by stack-level suppressions.
        _eks_internal_suppressions = [
            {"id": "AwsSolutions-IAM4", "reason": "CDK-internal EKS provider uses AWS managed policies (AWSLambdaBasicExecutionRole, AWSLambdaVPCAccessExecutionRole). Not user-configurable."},
            {"id": "AwsSolutions-IAM5", "reason": "CDK-internal EKS provider requires wildcard invoke permissions on its own Lambda functions for the custom resource framework. Not user-configurable."},
            {"id": "AwsSolutions-L1", "reason": "CDK-internal EKS provider Lambda runtime is managed by CDK library version. Not user-configurable."},
            {"id": "AwsSolutions-SF1", "reason": "CDK-internal EKS ClusterResourceProvider waiter state machine logging is managed by CDK. Not user-configurable."},
            {"id": "AwsSolutions-SF2", "reason": "CDK-internal EKS ClusterResourceProvider waiter state machine X-Ray tracing is managed by CDK. Not user-configurable."},
        ]
        for nested_construct in ["ClusterResourceProvider", "KubectlProvider"]:
            NagSuppressions.add_resource_suppressions_by_path(
                self,
                f"/{self.stack_name}/@aws-cdk--aws-eks.{nested_construct}",
                _eks_internal_suppressions,
                apply_to_children=True,
            )

        # ============================================================
        # Step Functions Orchestration (replaces setup.sh)
        # ============================================================

        # --- Target cluster overrides ---
        # When targeting an existing HyperPod cluster (e.g., riv-rig), override
        # the cluster name and EKS cluster name passed to the orchestration.
        target_cluster = self.node.try_get_context("target_cluster_name") or f"{project_tag}-cluster"
        target_eks = self.node.try_get_context("target_eks_cluster_name") or eks_cluster.cluster_name
        target_instance_type = self.node.try_get_context("target_instance_type") or instance_type
        target_instance_count = int(self.node.try_get_context("target_instance_count") or instance_count)

        # --- Training overrides (passed through pipeline event) ---
        training_method = self.node.try_get_context("training_method") or "RFT_MULTITURN_FULL"
        max_steps = str(self.node.try_get_context("max_steps") or "10")
        generation_replicas = str(self.node.try_get_context("generation_replicas") or "4")
        global_batch_size = str(self.node.try_get_context("global_batch_size") or "64")
        max_new_tokens = str(self.node.try_get_context("max_new_tokens") or "4096")
        max_length = str(self.node.try_get_context("max_length") or "16384")
        training_timeout = str(self.node.try_get_context("training_timeout") or "1800")

        # --- MLflow Tracking Server ---
        mlflow_tracking_server = sagemaker.CfnMlflowTrackingServer(
            self,
            "MlflowTrackingServer",
            tracking_server_name=f"{project_tag}-mlflow",
            role_arn=rft_role.role_arn,
            artifact_store_uri=f"s3://{training_bucket.bucket_name}/mlflow-artifacts/",
        )
        mlflow_tracking_arn = mlflow_tracking_server.attr_tracking_server_arn

        # MLflow permissions for roles that write metrics
        mlflow_policy = iam.ManagedPolicy(
            self,
            "MlflowPolicy",
            statements=[
                iam.PolicyStatement(
                    sid="MlflowTracking",
                    actions=[
                        "sagemaker-mlflow:AccessUI",
                        "sagemaker-mlflow:CreateExperiment",
                        "sagemaker-mlflow:CreateRun",
                        "sagemaker-mlflow:DeleteTag",
                        "sagemaker-mlflow:Get*",
                        "sagemaker-mlflow:LogBatch",
                        "sagemaker-mlflow:LogMetric",
                        "sagemaker-mlflow:LogParam",
                        "sagemaker-mlflow:Search*",
                        "sagemaker-mlflow:SetExperimentTag",
                        "sagemaker-mlflow:SetTag",
                        "sagemaker-mlflow:UpdateExperiment",
                        "sagemaker-mlflow:UpdateRun",
                    ],
                    resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:mlflow-tracking-server/*"],
                ),
                iam.PolicyStatement(
                    sid="MlflowPresignedUrl",
                    actions=["sagemaker:CreatePresignedMlflowTrackingServerUrl"],
                    resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:mlflow-tracking-server/*"],
                ),
            ],
        )
        NagSuppressions.add_resource_suppressions(
            mlflow_policy,
            [{"id": "AwsSolutions-IAM5", "reason": "MLflow tracking server name is dynamic; wildcard scoped to mlflow-tracking-server/* resource type."}],
        )
        rft_role.add_managed_policy(mlflow_policy)
        hyperpod_role.add_managed_policy(mlflow_policy)

        orchestration = TrainingOrchestration(
            self,
            "Orchestration",
            rft_role=rft_role,
            hyperpod_role=hyperpod_role,
            bucket_name=training_bucket.bucket_name,
            bucket_arn=training_bucket.bucket_arn,
            cluster_name=target_cluster,
            vpc_id=vpc.vpc_id,
            subnet_ids=",".join(private_subnet_ids),
            sg_id=training_sg.security_group_id,
            instance_type=target_instance_type,
            instance_count=target_instance_count,
            region=self.region,
            rft_role_arn=rft_role.role_arn,
            hyperpod_role_arn=hyperpod_role.role_arn,
            vf_env_id=vf_env_id,
            nova_model=nova_model,
            reward_cluster_arn=reward_cluster.cluster_arn,
            reward_cpu=reward_cpu,
            reward_memory=reward_memory,
            custom_env_s3_uri=custom_env_s3_uri,
            custom_env_id=custom_env_id,
            eks_cluster_name=target_eks,
            eks_cluster=eks_cluster,
            training_method=training_method,
            max_steps=max_steps,
            generation_replicas=generation_replicas,
            global_batch_size=global_batch_size,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            training_timeout=training_timeout,
            mlflow_tracking_arn=mlflow_tracking_arn,
        )

        cdk.CfnOutput(
            self,
            "StateMachineArn",
            value=orchestration.state_machine.state_machine_arn,
            description="Step Functions state machine ARN for training pipeline",
        )
        cdk.CfnOutput(
            self,
            "StateMachineName",
            value=orchestration.state_machine.state_machine_name,
            description="Step Functions state machine name",
        )
        cdk.CfnOutput(
            self,
            "MlflowTrackingServerArn",
            value=mlflow_tracking_arn,
            description="SageMaker MLflow tracking server ARN",
        )
