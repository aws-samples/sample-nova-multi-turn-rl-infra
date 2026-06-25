#!/bin/bash
# HyperPod lifecycle script with SDK install and S3 logging.
# All output is captured and uploaded to S3 for debugging since
# HyperPod doesn't pipe script stdout/stderr to CloudWatch.

set -euo pipefail
set -x

INSTANCE_ID=$(hostname)
BUCKET=$(echo "$SAGEMAKER_CLUSTER_LIFECYCLE_S3_URI" | sed 's|s3://||' | cut -d/ -f1)
LOG_KEY="lifecycle-logs/${INSTANCE_ID}.log"

# Capture all output to a log file AND stdout
exec > >(tee /tmp/lifecycle.log) 2>&1

echo "=========================================="
echo "HyperPod on_create lifecycle script"
echo "Instance: $INSTANCE_ID"
echo "Date: $(date -u)"
echo "Python: $(which python3) $(python3 --version 2>&1)"
echo "Pip: $(which pip) $(pip --version 2>&1)"
echo "Bucket: $BUCKET"
echo "=========================================="

echo "[1/4] Upgrading pip..."
pip install --upgrade pip

echo "[2/4] Installing Nova Forge SDK..."
pip install amzn-nova-forge==1.3.16

echo "[3/4] Verifying SDK installation..."
python3 -c "import amzn_nova_forge; print(f'SDK version: {amzn_nova_forge.__version__}')"

echo "[4/4] Lifecycle script complete."

# Upload log to S3 regardless of success/failure
if [ -n "$BUCKET" ]; then
    aws s3 cp /tmp/lifecycle.log "s3://$BUCKET/$LOG_KEY" 2>/dev/null || true
fi

echo "SUCCESS"
