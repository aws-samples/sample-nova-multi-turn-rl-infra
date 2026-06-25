# Shared Dockerfile for all Lambda functions that need the Nova Forge SDK.
# Each function is built by passing --build-arg FUNCTION_DIR=<function_name>.
#
# Usage (CDK handles this automatically via Code.from_docker_build):
#   docker build --build-arg FUNCTION_DIR=deploy_rft_infra -t deploy-rft .

FROM public.ecr.aws/lambda/python:3.12

ARG FUNCTION_DIR=deploy_rft_infra

# Install the Nova Forge SDK and its dependencies
RUN pip install --no-cache-dir amzn-nova-forge==1.3.16

# Copy shared utilities (used by all handlers)
COPY shared/ ${LAMBDA_TASK_ROOT}/shared/

# Copy the handler for this specific function
COPY ${FUNCTION_DIR}/handler.py ${LAMBDA_TASK_ROOT}/handler.py

# Run as non-root user for security (CKV_DOCKER_3)
# AWS Lambda base image provides the sbx_user1051 user
USER sbx_user1051

HEALTHCHECK NONE

CMD ["handler.handler"]
