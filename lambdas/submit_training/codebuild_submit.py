"""Step 4: Submit training job — executed by CodeBuild.

This script is invoked by the CodeBuild project in the Step Functions pipeline.
It receives the pipeline event as a JSON environment variable, configures
SMHPRuntimeManager and NovaModelCustomizer, and submits the training job.

CodeBuild is used instead of Lambda because SMHPRuntimeManager requires the
hyperpod CLI (kubectl + helm) which cannot run in Lambda containers.
"""
import json
import logging
import os
import sys

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_infra_kwargs(event):
    """Build RFTMultiturnInfrastructure kwargs from event — mirrors lambdas/shared/infra_utils.py."""
    sdk_resource_prefix = os.environ.get("SDK_RESOURCE_PREFIX", "nrl")
    kwargs = {
        "stack_name": event.get("sdk_stack_name", f"{sdk_resource_prefix}-{event.get('run_id', 'unknown')[:8]}"),
        "region": event["region"],
    }

    rft_role_arn = event.get("rft_role_arn", "")
    if rft_role_arn:
        kwargs["rft_role_name"] = rft_role_arn.split("/")[-1]

    custom_env_s3_uri = event.get("custom_env_s3_uri", "")
    if custom_env_s3_uri:
        from amzn_nova_forge.rft_multiturn.custom_environment import CustomEnvironment
        custom_env_id = event.get("custom_env_id", "")
        if not custom_env_id:
            raise ValueError("custom_env_id is required when custom_env_s3_uri is set")
        kwargs["custom_env"] = CustomEnvironment(env_id=custom_env_id, s3_uri=custom_env_s3_uri)
    else:
        from amzn_nova_forge.rft_multiturn.base_infra import VFEnvId
        vf_env_id = event["vf_env_id"]
        try:
            env_enum = VFEnvId(vf_env_id)
        except ValueError:
            valid = [e.value for e in VFEnvId]
            raise ValueError(f"Invalid vf_env_id '{vf_env_id}'. Must be one of: {valid}")
        kwargs["vf_env_id"] = env_enum

    reward_cluster_arn = event.get("reward_cluster_arn", "")
    if reward_cluster_arn:
        kwargs["infrastructure_arn"] = reward_cluster_arn
        subnet_ids = event.get("subnet_ids", "")
        security_group_id = event.get("security_group_id", "")
        if subnet_ids and security_group_id:
            kwargs["vpc_config"] = {
                "subnets": subnet_ids.split(","),
                "security_groups": [security_group_id],
            }
        kwargs["cpu"] = str(event.get("reward_cpu", "2048"))
        kwargs["memory"] = str(event.get("reward_memory", "4096"))
    else:
        kwargs["python_venv_name"] = event.get("python_venv_name", "rft_nova_venv")

    return kwargs


def main():
    from amzn_nova_forge.manager.runtime_manager import SMHPRuntimeManager
    from amzn_nova_forge.model import NovaModelCustomizer
    from amzn_nova_forge.model.model_enums import Model, TrainingMethod
    from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure

    pipeline_event_raw = os.environ.get("PIPELINE_EVENT")
    if not pipeline_event_raw:
        logger.error("PIPELINE_EVENT environment variable not set")
        sys.exit(1)

    event = json.loads(pipeline_event_raw)

    required = ("run_id", "cluster_name", "bucket_name", "instance_type",
                "instance_count", "sdk_stack_name", "region")
    missing = [k for k in required if k not in event]
    if missing:
        logger.error("Missing required fields in event: %s", missing)
        sys.exit(1)
    if not event.get("custom_env_s3_uri") and not event.get("vf_env_id"):
        logger.error("Either 'custom_env_s3_uri' or 'vf_env_id' must be provided")
        sys.exit(1)

    run_id = event["run_id"]
    cluster_name = event["cluster_name"]
    bucket_name = event["bucket_name"]
    instance_type = event["instance_type"]
    instance_count = int(event["instance_count"])
    region = event["region"]
    eks_cluster_name = event.get("eks_cluster_name", "")

    job_name = f"nova-rl-{run_id}"

    # Idempotency check
    sm = boto3.client("sagemaker", region_name=region)
    try:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        if status in ("InProgress", "Completed"):
            logger.info("Training job %s already exists (%s). Skipping.", job_name, status)
            logger.info(json.dumps({"job_name": job_name, "status": "SKIPPED_IDEMPOTENT"}))
            return
    except sm.exceptions.ClientError:
        pass

    nova_model_str = event.get("nova_model", "NOVA_LITE_2")
    model_enum = Model[nova_model_str]

    runtime = SMHPRuntimeManager(
        instance_type=instance_type,
        instance_count=instance_count,
        cluster_name=cluster_name,
        namespace="kubeflow",
    )

    training_method = event.get("training_method", "RFT_MULTITURN_FULL")
    method_enum = TrainingMethod[training_method]

    # MLflow monitoring
    mlflow_monitor = None
    mlflow_tracking_arn = event.get("mlflow_tracking_arn", "")
    if mlflow_tracking_arn:
        from amzn_nova_forge.monitor.mlflow_monitor import MLflowMonitor
        mlflow_monitor = MLflowMonitor(
            tracking_uri=mlflow_tracking_arn,
            experiment_name=f"nova-rl-{run_id}",
        )
        logger.info("MLflow tracking enabled: %s", mlflow_tracking_arn)

    customizer = NovaModelCustomizer(
        model=model_enum,
        method=method_enum,
        infra=runtime,
        data_s3_path=event.get("training_data_path", f"s3://{bucket_name}/training-data/training-data.jsonl"),
        output_s3_path=f"s3://{bucket_name}/output/",
        mlflow_monitor=mlflow_monitor,
    )

    infra_kwargs = build_infra_kwargs(event)
    rft_infra = RFTMultiturnInfrastructure(**infra_kwargs)
    rft_infra.setup()

    result = customizer.train(
        job_name=job_name,
        overrides={
            "max_steps": int(event.get("max_steps", 10)),
            "generation_replicas": int(event.get("generation_replicas", 4)),
            "max_new_tokens": int(event.get("max_new_tokens", 4096)),
            "max_length": int(event.get("max_length", 16384)),
            "global_batch_size": int(event.get("global_batch_size", 64)),
            "reasoning_effort": "null",
            "timeout": int(event.get("training_timeout", 1800)),
        },
        rft_multiturn_infra=rft_infra,
    )

    logger.info("Training job submitted: %s", job_name)
    logger.info("Output: s3://%s/output/%s/", bucket_name, job_name)


if __name__ == "__main__":
    main()
