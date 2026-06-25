#!/usr/bin/env python3
"""Nova Multi-Turn RL CDK Application.

Provisions AWS infrastructure for Amazon Nova model customization
using multi-turn reinforcement learning on SageMaker HyperPod.

Two-phase deployment:
  1. cdk deploy  — Provisions foundational infrastructure (VPC, IAM, S3,
     HyperPod, ECS, Step Functions, Lambdas, EventBridge)
  2. Upload a .jsonl file to S3 training-data/ prefix — EventBridge
     auto-triggers the Step Functions pipeline

setup.sh is available for manual/ad-hoc re-triggers.
"""
import os
import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks, NagReportFormat

from stacks.nova_rl_stack import NovaRlStack

app = cdk.App()

stack_name = app.node.try_get_context("stack_name") or "NovaMultiTurnRlStack"
region = app.node.try_get_context("region") or "us-east-1"
account = app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")

NovaRlStack(
    app,
    stack_name,
    env=cdk.Environment(account=account, region=region),
    description="Nova Multi-Turn RL infrastructure: HyperPod, S3, IAM, VPC",
)

# Apply AWS Solutions security checks with CSV report generation
cdk.Aspects.of(app).add(
    AwsSolutionsChecks(
        verbose=True,
        reports=True,
        report_formats=[NagReportFormat.CSV],
    )
)

app.synth()

