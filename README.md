# Nova Multi-Turn RL CDK Infrastructure

AWS infrastructure for Amazon Nova model customization using multi-turn reinforcement learning on SageMaker HyperPod.

## Architecture

Two-phase deployment:
1. `cdk deploy` provisions foundational infrastructure (VPC, IAM, S3, EKS, HyperPod, ECS, Step Functions, Lambdas, EventBridge).
2. Uploading a `.jsonl` file to the S3 `training-data/` prefix triggers the Step Functions pipeline via EventBridge.

```
EventBridge (S3 upload)
     │
     ▼
S3Trigger Lambda ──▶ Step Functions pipeline
                       │
                       ▼
   BootstrapEcrImage (CodeBuild)
     → DeployRftInfra (Lambda)
     → StartRewardWorkers (Lambda)
     → ValidateUploadData (Lambda)
     → SubmitTraining (CodeBuild, hyperpod CLI)
     → Success    (on failure → PipelineFailed → SNS alert)
```

Steps 0 and 4 run in CodeBuild (need Docker / hyperpod CLI). Steps 1–3 run in Docker-packaged Lambdas with the Nova SDK. Every step has idempotency checks, exponential-backoff retries, and a catch-all route to a failure state.

SDK-managed resources (conversation-proxy Lambdas, SQS FIFO queues, DynamoDB state table, ECS Fargate reward workers) are created at pipeline runtime, not by CDK.

## Prerequisites

- Python 3.12+, AWS CDK v2 (`npm install -g aws-cdk`), AWS CLI v2
- Docker or [Finch](https://github.com/runfinch/finch) for building Lambda images
- AWS account with `ml.p5.48xlarge` quota

## Quick Start

```bash
pip install -r requirements.txt

# Required: edit cdk.json and set project_tag + sdk_resource_prefix

cdk deploy --require-approval never

# Upload triggers the pipeline automatically
aws s3 cp training-data.jsonl s3://BUCKET_NAME/training-data/training-data.jsonl
```

Override parameters at deploy time:
```bash
cdk deploy -c instance_count=1 -c max_steps=20 -c global_batch_size=128
```

## Configuration

All parameters live in `cdk.json` under `context`.

**Infrastructure:** `project_tag` and `sdk_resource_prefix` are required prefixes. Key defaults: `instance_type=ml.p5.48xlarge`, `instance_count=10` (min 10 for `generation_replicas=4`), `nova_model=NOVA_LITE_2`, `region=us-east-1`, `eks_kubernetes_version=1.32`, `vf_env_id=wordle`.

**Training overrides** (flow through the pipeline event): `training_method`, `max_steps`, `generation_replicas`, `global_batch_size`, `max_new_tokens`, `max_length`, `training_timeout`.

**Target existing cluster:** set `target_cluster_name`, `target_eks_cluster_name`, `target_instance_type`, `target_instance_count` to run against a pre-existing HyperPod cluster.

### Training data format

Metadata-based JSONL, one record per line; `data_s3_path` must be a `.jsonl` file, not a directory:
```json
{"id": "wordle_train_001", "metadata": {"prompt": "Guess the 5-letter word", "answer": "crane"}}
```

### Custom reward environments

1. Create `custom-environments/<env-id>/` with a `load_environment()` function (see `my-custom-env`).
   `verifiers` is provided by the Nova Forge starter kit, so it does not need to be listed in the environment's dependencies.
2. Set `use_custom_env: "true"` and `custom_env_id: "<env-id>"` in `cdk.json`.
3. `cdk deploy` packages and uploads the environment to S3.

`client/nova_async_client.py` provides an OpenAI-compatible async client with SigV4 auth for custom reward functions that call the Nova inference endpoint.

## Monitoring

**Pipeline failure alerts** — subscribe to the SNS topic exported as `AlertTopicArn`:
```bash
aws sns subscribe --topic-arn <AlertTopicArn> \
  --protocol email --notification-endpoint you@example.com
```

**EventBridge trigger failures** — inspect the DLQ exported as `TriggerDlqUrl`.

**CloudWatch Logs:**
- `/aws/vendedlogs/states/nova-rl-*` — Step Functions execution
- `/aws/lambda/*{DeployRftInfra,StartRewardWorkers,ValidateUploadData}*` — steps 1–3
- `/aws/codebuild/nova-rl-submit-training` — step 4
- `/aws/sagemaker/Clusters/<cluster>/*` — HyperPod container logs

## Cleanup

1. Stop running Step Functions executions and ECS reward workers.
2. Delete SDK CloudFormation stacks (`<sdk_resource_prefix>-*-NovaForgeSDK`).
3. `cdk destroy`.

VPC deletion can fail due to GuardDuty-managed ENIs — delete VPC endpoints first, wait ~2 minutes, then retry.

## Security

- `cdk-nag` `AwsSolutionsChecks` enabled with documented suppressions
- S3 bucket: KMS encryption, block public access, SSL enforced, versioned
- KMS CMKs with rotation for CloudWatch Logs, SNS, EKS secrets, CodeBuild artifacts
- SNS and SQS DLQ enforce SSL
- IAM policies scoped to `sdk_resource_prefix` where SDK naming allows
- Lambda containers run as non-root
- VPC interface endpoints for ECR, EKS, STS, CloudWatch Logs, Lambda, SQS

**EKS public endpoint:** HyperPod requires a PUBLIC endpoint because its `/readyz` health check runs from AWS-managed service accounts outside the customer VPC. `PUBLIC_AND_PRIVATE` and private-only configurations cause cluster creation to fail. The upstream CIDR ranges are unpublished, so `public_access_cidrs` cannot be restricted. Access is protected by IAM authentication.

## Project Structure

```
app.py                        CDK app entry point
cdk.json                      CDK config and context defaults
requirements.txt
training-data.jsonl           Sample data (300 Wordle samples)
stacks/
  nova_rl_stack.py            VPC, EKS, HyperPod, S3, ECS, IAM
  orchestration.py            Step Functions, Lambdas, EventBridge
  iam_policies.py             Shared IAM policy builders
lambdas/
  Dockerfile                  Shared image for SDK Lambdas
  shared/infra_utils.py       RFT infrastructure kwargs builder
  deploy_rft_infra/           Step 1
  start_reward_workers/       Step 2
  validate_upload_data/       Step 3
  submit_training/            Step 4 (CodeBuild)
  s3_trigger/                 EventBridge → Step Functions
  helm_install_trigger/       Custom Resource for Helm install
lifecycle-scripts/on_create.sh  HyperPod node bootstrap
client/nova_async_client.py   Async client with SigV4 (BYOO)
custom-environments/          Custom reward environments
```

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `cdk deploy` fails on quota | Request `ml.p5.48xlarge` via Service Quotas |
| Upload doesn't trigger pipeline | Verify key matches `training-data/*.jsonl`; check `TriggerDlqUrl` |
| `data_s3_path` validation error | Path must end in `.jsonl` (not a directory) |
| `SubmitTraining` fails | Check CodeBuild logs; verify `hyperpod connect-cluster` succeeds |
| Training pods crash-loop | `/aws/sagemaker/Clusters/<cluster>/*` in CloudWatch |
| Reward workers idle | Check ECS task logs; confirm SQS queue depth |
| VPC deletion fails | Delete VPC endpoints, wait 2 min for GuardDuty ENIs |
| HyperPod `NotStabilized` error | EKS endpoint must be PUBLIC (not PUBLIC_AND_PRIVATE) |
