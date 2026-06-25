"""Shared utilities for building RFTMultiturnInfrastructure kwargs.

All Lambda handlers in the pipeline reconstruct the SDK's infrastructure
handle with identical parameters. This module centralizes that logic so
changes only need to happen in one place.
"""


def build_infra_kwargs(event):
    """Build RFTMultiturnInfrastructure constructor kwargs from event.

    Centralizes the ECS vs LOCAL mode logic and custom environment support.

    Environment resolution order:
      1. If custom_env_s3_uri is present → use CustomEnvironment (skip VFEnvId)
      2. Otherwise → validate vf_env_id against VFEnvId enum

    When reward_cluster_arn is present, the SDK runs reward workers on
    Fargate. Otherwise it falls back to LOCAL mode (only useful for
    local dev, not Lambda execution).
    """
    kwargs = {
        "stack_name": event.get("sdk_stack_name", f"nova-rft-infra-{event.get('run_id', 'unknown')}"),
        "region": event["region"],
    }

    # Pass the CDK-managed IAM role name so the SDK uses the same role
    # across all pipeline steps. Without this, the SDK defaults to
    # "RFTExecutionRoleNovaSDK" which may not match the CDK-created role.
    rft_role_arn = event.get("rft_role_arn", "")
    if rft_role_arn:
        kwargs["rft_role_name"] = rft_role_arn.split("/")[-1]

    # --- Environment resolution: custom env vs built-in VFEnvId ---
    custom_env_s3_uri = event.get("custom_env_s3_uri", "")
    if custom_env_s3_uri:
        # Custom environment mode: construct a CustomEnvironment with the
        # pre-uploaded S3 tarball URI. The SDK attaches this to infra via
        # self.infra.custom_env and passes env_id to _build_setup_commands.
        from amzn_nova_forge.rft_multiturn.custom_environment import CustomEnvironment

        custom_env_id = event.get("custom_env_id", "")
        if not custom_env_id:
            raise ValueError("custom_env_id is required when custom_env_s3_uri is set")

        kwargs["custom_env"] = CustomEnvironment(
            env_id=custom_env_id,
            s3_uri=custom_env_s3_uri,
        )
    else:
        # Built-in environment mode: validate against the VFEnvId enum
        from amzn_nova_forge.rft_multiturn.base_infra import VFEnvId

        vf_env_id = event["vf_env_id"]
        try:
            env_enum = VFEnvId(vf_env_id)
        except ValueError:
            valid = [e.value for e in VFEnvId]
            raise ValueError(f"Invalid vf_env_id '{vf_env_id}'. Must be one of: {valid}")

        kwargs["vf_env_id"] = env_enum

    # --- Infrastructure mode: ECS (production) vs LOCAL (dev only) ---
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

        cpu = event.get("reward_cpu", "2048")
        memory = event.get("reward_memory", "4096")
        kwargs["cpu"] = cpu
        kwargs["memory"] = memory
    else:
        # LOCAL mode fallback (dev only — broken inside Lambda containers)
        kwargs["python_venv_name"] = event.get("python_venv_name", "rft_nova_venv")

    return kwargs
