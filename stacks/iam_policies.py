"""Shared IAM policy builders for Nova Multi-Turn RL stack.

Centralizes policies that are identical across nova_rl_stack.py and
orchestration.py to avoid drift. Policies with role-specific resource
scoping remain inline in their respective files.
"""
import aws_cdk as cdk
from aws_cdk import aws_iam as iam
from cdk_nag import NagSuppressions
from constructs import Construct


def create_ecs_management_policy(
    scope: Construct,
    construct_id: str,
) -> iam.ManagedPolicy:
    """ECS cluster and task management policy.

    Uses wildcard resources because CreateCluster, ListClusters, and
    DescribeTaskDefinition don't support resource-level permissions.
    """
    policy = iam.ManagedPolicy(
        scope,
        construct_id,
        description="ECS cluster and task management",
        statements=[
            iam.PolicyStatement(
                sid="EcsClusterManagement",
                actions=[
                    "ecs:CreateCluster",
                    "ecs:DeleteCluster",
                    "ecs:DescribeClusters",
                    "ecs:ListClusters",
                ],
                resources=["*"],
            ),
            iam.PolicyStatement(
                sid="EcsTaskManagement",
                actions=[
                    "ecs:RegisterTaskDefinition",
                    "ecs:DeregisterTaskDefinition",
                    "ecs:DescribeTaskDefinition",
                    "ecs:RunTask",
                    "ecs:StopTask",
                    "ecs:DescribeTasks",
                    "ecs:ListTasks",
                    "ecs:CreateService",
                    "ecs:UpdateService",
                    "ecs:DeleteService",
                    "ecs:DescribeServices",
                ],
                resources=["*"],
            ),
        ],
    )
    NagSuppressions.add_resource_suppressions(
        policy,
        [
            {
                "id": "AwsSolutions-IAM5",
                "reason": "ecs:CreateCluster, ecs:ListClusters, and ecs:DescribeTaskDefinition do not support resource-level permissions (AWS service limitation).",
            }
        ],
    )
    return policy


def create_cloudwatch_logs_policy(
    scope: Construct,
    construct_id: str,
    region: str,
    log_group_prefix: str = "/aws/lambda/*",
) -> iam.ManagedPolicy:
    """CloudWatch Logs policy scoped to a log group prefix."""
    policy = iam.ManagedPolicy(
        scope,
        construct_id,
        description=f"CloudWatch Logs for {log_group_prefix}",
        statements=[
            iam.PolicyStatement(
                sid="CloudWatchLogsAccess",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{region}:{cdk.Aws.ACCOUNT_ID}:log-group:{log_group_prefix}",
                    f"arn:aws:logs:{region}:{cdk.Aws.ACCOUNT_ID}:log-group:{log_group_prefix}:*",
                ],
            )
        ],
    )
    NagSuppressions.add_resource_suppressions(
        policy,
        [
            {
                "id": "AwsSolutions-IAM5",
                "reason": f"CloudWatch Logs wildcard scoped to {log_group_prefix} for dynamically named log groups.",
            }
        ],
    )
    return policy
