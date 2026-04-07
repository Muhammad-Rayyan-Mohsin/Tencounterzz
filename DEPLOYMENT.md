# TenCount Deployment Guide

## Prerequisites

- AWS CLI installed and configured (`brew install awscli && aws configure`)
- AWS IAM user with `AdministratorAccess` policy
- Region: `eu-north-1` (Stockholm)

## Quick Start

```bash
# Deploy (launches EC2 instance, uploads code, installs deps, starts app)
./deploy.sh deploy

# Your app will be available at http://<PUBLIC_IP>:3000
```

## Management Commands

```bash
./deploy.sh status    # Show instance state and URL
./deploy.sh stop      # Stop instance (no charges while stopped, data preserved)
./deploy.sh start     # Start a stopped instance
./deploy.sh ssh       # SSH into the server
./deploy.sh teardown  # Terminate instance and delete security group
```

## Viewing Logs

```bash
./deploy.sh ssh
sudo journalctl -u tencount -f    # Live app logs
```

## Instance Details

- **Current type**: `m7i-flex.large` (2 vCPUs, 8GB RAM, CPU-only)
- **Cost**: ~$0.10/hr while running, $0 while stopped
- **OS**: Ubuntu 24.04
- **Storage**: 100GB gp3 EBS volume

## Upgrading to GPU (Faster Inference)

GPU instances require a quota increase (0 by default on new AWS accounts).

1. Request quota in AWS Console: **Service Quotas > EC2 > Running On-Demand G and VT instances** — set to 4
2. Wait for approval (24-48hrs, check email)
3. Deploy with GPU:

```bash
./deploy.sh teardown
TENCOUNT_INSTANCE_TYPE=g4dn.xlarge TENCOUNT_AMI_ID=ami-0248a5203d01dc336 ./deploy.sh deploy
```

The GPU AMI (Deep Learning OSS Nvidia Driver AMI) includes PyTorch with CUDA pre-installed.

## Environment Variables

These are set automatically by the deploy script via systemd:

| Variable | Default (deployed) | Purpose |
|----------|-------------------|---------|
| `PYTHON_BIN` | `/home/ubuntu/tencount/.venv/bin/python3` | Python interpreter path |
| `PROJECT_ROOT` | `/home/ubuntu/tencount` | Project root for model path resolution |
| `NODE_ENV` | `production` | Next.js production mode |

## What Gets Deployed

- Frontend (Next.js 14) — built and served via `next start` on port 3000
- Python inference pipeline — PyTorch + YOLO + BiLSTM punch classifier
- Model weights (~78MB): YOLOv11m detection, YOLOv8m pose, AttentionBiLSTM classifier

## Redeploying After Code Changes

```bash
# If instance is running, teardown first
./deploy.sh teardown

# Then redeploy
./deploy.sh deploy
```

## Troubleshooting

**App not loading**: Check if the service is running:
```bash
./deploy.sh ssh
sudo systemctl status tencount
```

**Inference fails with missing module**: SSH in and install it:
```bash
./deploy.sh ssh
source ~/tencount/.venv/bin/activate
pip install <missing-package>
```

**Upload times out**: Large videos may fail on slow connections. Try a smaller file (<50MB).

**SSH key issues**: The key is stored at `~/.ssh/tencount-deploy-key.pem`. If deleted, teardown and redeploy.
