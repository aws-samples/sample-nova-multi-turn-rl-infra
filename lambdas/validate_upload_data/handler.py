"""Step 3: Validate and upload training data to S3.

Uses JSONLDatasetLoader to validate JSONL format compliance
and upload to the S3 training data prefix.
"""
import logging
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    """Validate and upload training data."""
    from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader

    if "bucket_name" not in event:
        raise ValueError("Missing required field in event: ['bucket_name']")

    bucket_name = event["bucket_name"]
    training_data_path = event.get("training_data_path", "")
    s3_dest = f"s3://{bucket_name}/training-data/"

    if not training_data_path:
        logger.info("No training_data_path provided. Skipping upload.")
        return {
            **event,
            "training_data_s3": s3_dest,
            "step": "validate_upload_data",
            "status": "SKIPPED_NO_DATA",
        }

    # Idempotency: skip upload only when the destination key exists AND its
    # content matches the source (ETag comparison). Same filename with different
    # content triggers a re-upload instead of being silently skipped.
    import os
    filename = os.path.basename(training_data_path.replace("s3://", "").split("/", 1)[-1])
    target_key = f"training-data/{filename}"

    s3 = boto3.client("s3")
    try:
        dest_head = s3.head_object(Bucket=bucket_name, Key=target_key)
        dest_etag = dest_head.get("ETag", "")

        # If source is also in S3, compare ETags directly
        source_etag = ""
        if training_data_path.startswith("s3://"):
            src_parts = training_data_path.replace("s3://", "").split("/", 1)
            src_bucket, src_key = src_parts[0], src_parts[1]
            src_head = s3.head_object(Bucket=src_bucket, Key=src_key)
            source_etag = src_head.get("ETag", "")

        if source_etag and source_etag == dest_etag:
            logger.info("File s3://%s/%s matches source ETag. Skipping upload.", bucket_name, target_key)
            return {
                **event,
                "training_data_s3": f"s3://{bucket_name}/{target_key}",
                "step": "validate_upload_data",
                "status": "SKIPPED_IDEMPOTENT",
            }
        elif source_etag:
            logger.info("File exists but ETag differs (src=%s, dest=%s). Re-uploading.", source_etag, dest_etag)
        # If source is local (no ETag to compare), always re-upload when content may differ
    except s3.exceptions.ClientError:
        pass  # File doesn't exist at destination, proceed with upload

    loader = JSONLDatasetLoader(id="id", metadata="metadata")
    loader.load(training_data_path)

    # Upload validated data to S3 using the SDK's save_data method
    s3_dest_path = f"s3://{bucket_name}/{target_key}"
    loader.save_data(s3_dest_path)

    return {
        **event,
        "training_data_s3": f"s3://{bucket_name}/{target_key}",
        "step": "validate_upload_data",
        "status": "SUCCESS",
    }
