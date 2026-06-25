"""
Custom Resource Lambda handler for triggering HyperPod Helm chart installation.

This Lambda function is invoked by CloudFormation as a Custom Resource to trigger
a CodeBuild project that installs the HyperPod Helm chart on the EKS cluster.
It polls the build status every 15 seconds until completion.
"""
import boto3
import json
import time
import urllib.request


def send(event, context, status, data=None, reason=""):
    """Send response back to CloudFormation."""
    body = json.dumps({
        "Status": status,
        "Reason": reason or f"See logs: {context.log_stream_name}",
        "PhysicalResourceId": context.log_stream_name,
        "StackId": event["StackId"],
        "RequestId": event["RequestId"],
        "LogicalResourceId": event["LogicalResourceId"],
        "Data": data or {},
    }).encode()
    urllib.request.urlopen(urllib.request.Request(  # nosec B310 # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        event["ResponseURL"], data=body, headers={"Content-Type": ""}, method="PUT"))


def handler(event, context):
    """
    Handle CloudFormation Custom Resource lifecycle events.
    
    On Create/Update: Triggers CodeBuild project and polls for completion.
    On Delete: Returns success immediately (Helm chart cleanup handled separately).
    """
    try:
        if event.get("RequestType") == "Delete":
            send(event, context, "SUCCESS")
            return
        
        cb = boto3.client("codebuild")
        build_id = cb.start_build(
            projectName=event["ResourceProperties"]["ProjectName"]
        )["build"]["id"]
        
        while True:
            time.sleep(15)
            status = cb.batch_get_builds(ids=[build_id])["builds"][0]["buildStatus"]
            if status == "SUCCEEDED":
                send(event, context, "SUCCESS", {"BuildId": build_id})
                return
            if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
                send(event, context, "FAILED", reason=f"Build {status}: {build_id}")
                return
    except Exception as e:
        send(event, context, "FAILED", reason=str(e))
