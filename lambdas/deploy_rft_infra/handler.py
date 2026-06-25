"""Step 1: Deploy RFT multi-turn infrastructure.

Creates the SDK's CloudFormation stack with Lambda functions,
SQS FIFO queues, and DynamoDB table for multi-turn conversation management.

The SDK's RFTMultiturnInfrastructure is configured in ECS mode
(infrastructure_arn points to a CDK-managed ECS cluster) so reward
workers run on Fargate instead of inside the Lambda container.
"""
import logging
import os
import boto3

from shared.infra_utils import build_infra_kwargs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    """Deploy RFT infrastructure via Nova Forge SDK."""
    from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure

    missing = [k for k in ("run_id", "region", "rft_role_arn") if k not in event]
    if missing:
        raise ValueError(f"Missing required fields in event: {missing}")
    # Environment is required but can be either custom_env_s3_uri or vf_env_id
    if not event.get("custom_env_s3_uri") and not event.get("vf_env_id"):
        raise ValueError("Either 'custom_env_s3_uri' or 'vf_env_id' must be provided")

    run_id = event["run_id"]
    region = event["region"]
    sdk_resource_prefix = os.environ.get("SDK_RESOURCE_PREFIX", "nrl")
    sdk_stack_name = f"{sdk_resource_prefix}-{run_id[:8]}"

    # Inject sdk_stack_name into event for build_infra_kwargs
    event_with_stack = {**event, "sdk_stack_name": sdk_stack_name}

    # Check idempotency — skip if SDK stack already exists.
    # The SDK appends "-NovaForgeSDK" suffix to the stack name internally
    # (see base_infra.py STACK_NAME_SUFFIX), so check both forms.
    cf = boto3.client("cloudformation", region_name=region)
    for candidate in (sdk_stack_name, f"{sdk_stack_name}-NovaForgeSDK"):
        try:
            resp = cf.describe_stacks(StackName=candidate)
            status = resp["Stacks"][0]["StackStatus"]
            if status in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
                logger.info("SDK stack %s already exists (%s). Skipping.", candidate, status)
                return {
                    **event,
                    "sdk_stack_name": sdk_stack_name,
                    "step": "deploy_rft_infra",
                    "status": "SKIPPED_IDEMPOTENT",
                }
        except cf.exceptions.ClientError:
            pass  # Stack doesn't exist under this name

    infra_kwargs = build_infra_kwargs(event_with_stack)
    rft_infra = RFTMultiturnInfrastructure(**infra_kwargs)
    rft_infra.setup()

    return {
        **event,
        "sdk_stack_name": sdk_stack_name,
        "step": "deploy_rft_infra",
        "status": "SUCCESS",
    }
