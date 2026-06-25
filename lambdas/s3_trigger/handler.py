"""S3 event to Step Functions trigger Lambda.

Transforms S3 EventBridge events into Step Functions pipeline input
and starts execution. Provides deduplication based on S3 key hash.
"""
import hashlib
import json
import logging
import os
import boto3
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sfn = boto3.client("stepfunctions")


def handler(event, context):
    """Transform S3 EventBridge event into Step Functions input and start execution."""
    detail = event.get("detail", {})
    s3_key = detail.get("object", {}).get("key", "")

    # Only trigger on .jsonl files
    if not s3_key.endswith(".jsonl"):
        logger.info("Ignoring non-JSONL upload: %s", s3_key)
        return {"status": "SKIPPED", "reason": "not a .jsonl file"}

    # Derive execution name from the S3 key so uploading the same file
    # twice is deduplicated by Step Functions (ExecutionAlreadyExists).
    # Different files get different hashes and run as separate pipelines.
    key_hash = hashlib.md5(s3_key.encode(), usedforsecurity=False).hexdigest()[:8]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + key_hash
    execution_name = f"s3-trigger-{key_hash}"

    bucket = os.environ["BUCKET_NAME"]

    pipeline_input = {
        "run_id": run_id,
        "region": os.environ["REGION"],
        "cluster_name": os.environ["CLUSTER_NAME"],
        "bucket_name": bucket,
        "hyperpod_role_arn": os.environ["HYPERPOD_ROLE_ARN"],
        "rft_role_arn": os.environ["RFT_ROLE_ARN"],
        "instance_type": os.environ["INSTANCE_TYPE"],
        "instance_count": os.environ["INSTANCE_COUNT"],
        "training_data_path": f"s3://{bucket}/{s3_key}",
        "vf_env_id": os.environ.get("VF_ENV_ID", "wordle"),
        "nova_model": os.environ.get("NOVA_MODEL", "NOVA_LITE_2"),
        "reward_cluster_arn": os.environ.get("REWARD_CLUSTER_ARN", ""),
        "reward_cpu": os.environ.get("REWARD_CPU", "2048"),
        "reward_memory": os.environ.get("REWARD_MEMORY", "4096"),
        "subnet_ids": os.environ.get("SUBNET_IDS", ""),
        "security_group_id": os.environ.get("SECURITY_GROUP_ID", ""),
        "custom_env_s3_uri": os.environ.get("CUSTOM_ENV_S3_URI", ""),
        "custom_env_id": os.environ.get("CUSTOM_ENV_ID", ""),
        "eks_cluster_name": os.environ.get("EKS_CLUSTER_NAME", ""),
        "training_method": os.environ.get("TRAINING_METHOD", "RFT_MULTITURN_FULL"),
        "max_steps": os.environ.get("MAX_STEPS", "10"),
        "generation_replicas": os.environ.get("GENERATION_REPLICAS", "4"),
        "global_batch_size": os.environ.get("GLOBAL_BATCH_SIZE", "64"),
        "max_new_tokens": os.environ.get("MAX_NEW_TOKENS", "4096"),
        "max_length": os.environ.get("MAX_LENGTH", "16384"),
        "training_timeout": os.environ.get("TRAINING_TIMEOUT", "1800"),
        "mlflow_tracking_arn": os.environ.get("MLFLOW_TRACKING_ARN", ""),
    }

    try:
        resp = sfn.start_execution(
            stateMachineArn=os.environ["STATE_MACHINE_ARN"],
            name=execution_name,
            input=json.dumps(pipeline_input),
        )
        logger.info("Started execution: %s", resp['executionArn'])
        return {"status": "STARTED", "executionArn": resp["executionArn"]}
    except sfn.exceptions.ExecutionAlreadyExists:
        logger.info("Execution already exists for %s (name=%s). Skipping duplicate.", s3_key, execution_name)
        return {"status": "DEDUPLICATED", "reason": f"execution {execution_name} already running"}
