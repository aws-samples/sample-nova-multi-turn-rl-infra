"""Step 2: Start reward workers on ECS Fargate.

Deploys Fargate tasks running the reward environment
for multi-turn RL conversation evaluation.

When reward_cluster_arn is present in the event, the SDK uses the
CDK-managed ECS cluster (production mode). The idempotency check
looks for running tasks in that specific cluster.
"""
import logging
import boto3

from shared.infra_utils import build_infra_kwargs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    """Start reward workers via Nova Forge SDK."""
    from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure

    missing = [k for k in ("sdk_stack_name", "region") if k not in event]
    if missing:
        raise ValueError(f"Missing required fields in event: {missing}")
    if not event.get("custom_env_s3_uri") and not event.get("vf_env_id"):
        raise ValueError("Either 'custom_env_s3_uri' or 'vf_env_id' must be provided")

    sdk_stack_name = event["sdk_stack_name"]
    region = event["region"]

    # Determine which ECS cluster to check for idempotency.
    # In ECS mode, the cluster is CDK-managed (passed via event).
    # In LOCAL mode fallback, check the SDK's CloudFormation stack.
    reward_cluster_arn = event.get("reward_cluster_arn", "")
    ecs_cluster_arn = reward_cluster_arn or _get_sdk_ecs_cluster(sdk_stack_name, region)

    if ecs_cluster_arn:
        ecs = boto3.client("ecs", region_name=region)
        try:
            # Filter by task definition family so stale tasks from crashed
            # runs or unrelated services don't cause false-positive skips.
            # The SDK names its task def family after the stack name.
            task_family = event.get("task_family", sdk_stack_name)
            running = ecs.list_tasks(
                cluster=ecs_cluster_arn,
                desiredStatus="RUNNING",
                family=task_family,
            )
            if running.get("taskArns"):
                logger.info(
                    "Reward workers already running in %s (family=%s, count=%d). Skipping.",
                    ecs_cluster_arn, task_family, len(running['taskArns']),
                )
                return {
                    **event,
                    "ecs_cluster_arn": ecs_cluster_arn,
                    "step": "start_reward_workers",
                    "status": "SKIPPED_IDEMPOTENT",
                }
        except Exception as e:
            logger.warning("ECS idempotency check failed: %s. Proceeding with creation.", e)

    infra_kwargs = build_infra_kwargs(event)
    rft_infra = RFTMultiturnInfrastructure(**infra_kwargs)
    # setup() is idempotent — on an existing stack it just loads outputs
    # without redeploying. Required because the constructor doesn't populate
    # stack_outputs, and start_training_environment() needs them.
    rft_infra.setup()
    rft_infra.start_training_environment(vf_env_args={})

    return {
        **event,
        "ecs_cluster_arn": ecs_cluster_arn or "unknown",
        "step": "start_reward_workers",
        "status": "SUCCESS",
    }


def _get_sdk_ecs_cluster(sdk_stack_name: str, region: str) -> str | None:
    """Find the ECS cluster ARN from the SDK's CloudFormation stack outputs."""
    cf = boto3.client("cloudformation", region_name=region)
    try:
        resp = cf.describe_stack_resources(
            StackName=sdk_stack_name,
        )
        for resource in resp.get("StackResources", []):
            if resource["ResourceType"] == "AWS::ECS::Cluster":
                return resource["PhysicalResourceId"]
    except Exception as e:
        logger.warning("Failed to lookup ECS cluster from stack %s: %s", sdk_stack_name, e)
    return None
