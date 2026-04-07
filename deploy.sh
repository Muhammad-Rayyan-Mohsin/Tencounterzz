#!/usr/bin/env bash
set -euo pipefail

# ── TenCount AWS Deployment Script ──────────────────────────────────────────
# Usage:
#   ./deploy.sh deploy   — Launch EC2 GPU instance & deploy app
#   ./deploy.sh start    — Start a stopped instance
#   ./deploy.sh stop     — Stop the instance (saves money, keeps data)
#   ./deploy.sh status   — Show instance status & URL
#   ./deploy.sh ssh      — SSH into the instance
#   ./deploy.sh teardown — Terminate instance & clean up all AWS resources
# ────────────────────────────────────────────────────────────────────────────

REGION="eu-north-1"
# Switch to g4dn.xlarge + Deep Learning AMI once GPU quota is approved
INSTANCE_TYPE="${TENCOUNT_INSTANCE_TYPE:-m7i-flex.large}"
AMI_ID="${TENCOUNT_AMI_ID:-ami-0dab98137e5c11cb8}"  # Ubuntu 24.04 (CPU) or Deep Learning AMI (GPU)
KEY_NAME="tencount-deploy-key"
SG_NAME="tencount-sg"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
STATE_FILE="$(dirname "$0")/.deploy-state"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Helpers ────────────────────────────────────────────────────────────────

red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
blue()  { printf '\033[0;34m%s\033[0m\n' "$*"; }

get_instance_id() {
  [[ -f "$STATE_FILE" ]] && cat "$STATE_FILE" || echo ""
}

get_public_ip() {
  local iid="$1"
  aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null
}

get_instance_state() {
  local iid="$1"
  aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null
}

wait_for_ssh() {
  local ip="$1"
  blue "Waiting for SSH to be ready on $ip..."
  for i in $(seq 1 60); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" ubuntu@"$ip" "echo ok" &>/dev/null; then
      green "SSH ready!"
      return 0
    fi
    printf "."
    sleep 5
  done
  red "SSH timeout after 5 minutes"
  return 1
}

remote() {
  local ip="$1"; shift
  ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$ip" "$@"
}

# ── Deploy ─────────────────────────────────────────────────────────────────

do_deploy() {
  blue "=== TenCount GPU Deployment ==="
  echo "  Region: $REGION | Instance: $INSTANCE_TYPE | AMI: Deep Learning PyTorch 2.10"
  echo ""

  # 1. Key pair
  if [[ ! -f "$KEY_FILE" ]]; then
    blue "Creating key pair..."
    aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
      --query 'KeyMaterial' --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    green "Key saved to $KEY_FILE"
  else
    blue "Key pair already exists at $KEY_FILE"
  fi

  # 2. Security group
  SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

  if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    blue "Creating security group..."
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
      --group-name "$SG_NAME" \
      --description "TenCount app - SSH, HTTP, HTTPS, Next.js" \
      --query 'GroupId' --output text)

    # Allow SSH, HTTP, HTTPS, Next.js port
    for port in 22 80 443 3000; do
      aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" --protocol tcp --port "$port" --cidr 0.0.0.0/0 2>/dev/null || true
    done
    green "Security group created: $SG_ID"
  else
    blue "Security group already exists: $SG_ID"
  fi

  # 3. Launch instance
  blue "Launching $INSTANCE_TYPE instance..."
  INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=TenCount-FYP}]" \
    --query 'Instances[0].InstanceId' --output text)

  echo "$INSTANCE_ID" > "$STATE_FILE"
  green "Instance launched: $INSTANCE_ID"

  # 4. Wait for running
  blue "Waiting for instance to start..."
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
  green "Instance is running"

  PUBLIC_IP=$(get_public_ip "$INSTANCE_ID")
  blue "Public IP: $PUBLIC_IP"

  # 5. Wait for SSH
  wait_for_ssh "$PUBLIC_IP"

  # 6. Upload project files
  blue "Uploading project files (~80MB)..."
  cd "$PROJECT_DIR"
  tar czf /tmp/tencount-deploy.tar.gz \
    --exclude='.git' \
    --exclude='frontend/node_modules' \
    --exclude='frontend/.next' \
    --exclude='.DS_Store' \
    --exclude='__pycache__' \
    --exclude='.deploy-state' \
    .
  scp -o StrictHostKeyChecking=no -i "$KEY_FILE" /tmp/tencount-deploy.tar.gz ubuntu@"$PUBLIC_IP":/tmp/
  rm /tmp/tencount-deploy.tar.gz
  green "Upload complete"

  # 7. Setup on server
  blue "Setting up server (installing Node.js, Python deps, building frontend)..."
  remote "$PUBLIC_IP" bash -s <<'SETUP_SCRIPT'
set -e

# Extract project
mkdir -p ~/tencount
cd ~/tencount
tar xzf /tmp/tencount-deploy.tar.gz
rm /tmp/tencount-deploy.tar.gz

# System deps
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv libgl1 libglib2.0-0 > /dev/null

# Install Node.js 20
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

# Python venv + deps (CPU PyTorch for now, switch to CUDA when GPU available)
python3 -m venv ~/tencount/.venv
source ~/tencount/.venv/bin/activate
pip install --upgrade pip > /dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null
pip install ultralytics opencv-python-headless scipy numpy > /dev/null

# Build Next.js frontend
cd ~/tencount/frontend
npm install --production=false
npm run build

# Create systemd service for the app
sudo tee /etc/systemd/system/tencount.service > /dev/null <<EOF
[Unit]
Description=TenCount Boxing Analytics
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tencount/frontend
Environment=NODE_ENV=production
Environment=PYTHON_BIN=/home/ubuntu/tencount/.venv/bin/python3
Environment=PROJECT_ROOT=/home/ubuntu/tencount
Environment=PORT=3000
ExecStart=/usr/bin/node /home/ubuntu/tencount/frontend/node_modules/.bin/next start -p 3000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tencount
sudo systemctl start tencount

echo "SETUP_COMPLETE"
SETUP_SCRIPT

  green ""
  green "============================================"
  green "  TenCount deployed successfully!"
  green "============================================"
  green ""
  green "  URL:  http://$PUBLIC_IP:3000"
  green "  SSH:  ./deploy.sh ssh"
  green ""
  green "  Stop:  ./deploy.sh stop   (saves \$\$, keeps data)"
  green "  Start: ./deploy.sh start  (resumes)"
  green ""
  blue "  Cost: ~\$0.53/hr while running. Stop when not testing!"
  echo ""
}

# ── Start ──────────────────────────────────────────────────────────────────

do_start() {
  local iid=$(get_instance_id)
  [[ -z "$iid" ]] && red "No instance found. Run: ./deploy.sh deploy" && exit 1

  blue "Starting instance $iid..."
  aws ec2 start-instances --region "$REGION" --instance-ids "$iid" > /dev/null
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$iid"

  PUBLIC_IP=$(get_public_ip "$iid")
  wait_for_ssh "$PUBLIC_IP"

  # Restart the app service
  remote "$PUBLIC_IP" "sudo systemctl start tencount"

  green ""
  green "Instance started! URL: http://$PUBLIC_IP:3000"
  blue "Remember to stop when done: ./deploy.sh stop"
}

# ── Stop ───────────────────────────────────────────────────────────────────

do_stop() {
  local iid=$(get_instance_id)
  [[ -z "$iid" ]] && red "No instance found." && exit 1

  blue "Stopping instance $iid (data preserved, no charges while stopped)..."
  aws ec2 stop-instances --region "$REGION" --instance-ids "$iid" > /dev/null
  green "Instance stopping. Run './deploy.sh start' to resume."
}

# ── Status ─────────────────────────────────────────────────────────────────

do_status() {
  local iid=$(get_instance_id)
  [[ -z "$iid" ]] && red "No instance found. Run: ./deploy.sh deploy" && exit 1

  local state=$(get_instance_state "$iid")
  local ip=$(get_public_ip "$iid")

  echo ""
  echo "  Instance: $iid"
  echo "  State:    $state"
  if [[ "$state" == "running" && "$ip" != "None" ]]; then
    green "  URL:      http://$ip:3000"
  fi
  echo ""
}

# ── SSH ────────────────────────────────────────────────────────────────────

do_ssh() {
  local iid=$(get_instance_id)
  [[ -z "$iid" ]] && red "No instance found." && exit 1

  local ip=$(get_public_ip "$iid")
  [[ "$ip" == "None" ]] && red "Instance not running. Run: ./deploy.sh start" && exit 1

  ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$ip"
}

# ── Teardown ───────────────────────────────────────────────────────────────

do_teardown() {
  local iid=$(get_instance_id)
  if [[ -n "$iid" ]]; then
    blue "Terminating instance $iid..."
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$iid" > /dev/null
    rm -f "$STATE_FILE"
    green "Instance terminated."
  fi

  # Clean up security group (wait for instance to fully terminate)
  blue "Cleaning up security group..."
  sleep 5
  SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
  if [[ "$SG_ID" != "None" && -n "$SG_ID" ]]; then
    aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" 2>/dev/null || \
      blue "Security group still in use — will be deleted once instance terminates"
  fi

  green "Teardown complete. Key pair kept at $KEY_FILE for future deploys."
}

# ── Main ───────────────────────────────────────────────────────────────────

case "${1:-}" in
  deploy)   do_deploy ;;
  start)    do_start ;;
  stop)     do_stop ;;
  status)   do_status ;;
  ssh)      do_ssh ;;
  teardown) do_teardown ;;
  *)
    echo "Usage: ./deploy.sh {deploy|start|stop|status|ssh|teardown}"
    exit 1
    ;;
esac
