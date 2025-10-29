# GitHub Actions Self-Hosted Runner Setup

This directory contains automated scripts to deploy GitHub Actions self-hosted runners for the stock-prediction repository. Self-hosted runners allow you to run CI/CD workflows on your own infrastructure, providing better control over the environment and access to local resources.

## Deployment Options

### 1. Bare-Metal with systemd (Recommended)
Runs directly on the host with systemd service management. Best for production environments.

### 2. Docker Container (Alternative)
Runs in an isolated container. Good for testing or when you need isolation.

⚠️ **Security Note**: Never commit runner tokens or secrets to the repository. Store them in:
- `secrets/` directory (gitignored)
- Environment variables
- GitHub Actions Secrets

## Prerequisites

### Required
- Linux host (Ubuntu 22.04+ or Debian 11+)
- System packages: `curl`, `tar`, `systemd`
- Sufficient disk space: 10GB minimum for runner + work directory
- Network access to GitHub

### Optional
- Docker and Docker Compose (for containerized deployment)
- sudo access (for systemd service installation)

## Runner Labels

Runners are tagged with labels that workflows use to target specific runners:

| Label | Description | Auto-added |
|-------|-------------|------------|
| `self-hosted` | All self-hosted runners | Yes |
| `stock-ci` | Stock-prediction runner | Default |
| `linux` | Linux OS runner | Default |
| `gpu` | Has GPU access | Default |
| `large` | High-resource runner | Manual |

Customize labels using the `--labels` parameter during registration.

## Quick Start Guide

### Step 1: Get Registration Token

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Actions** → **Runners**
3. Click **"New self-hosted runner"**
4. Select **Linux**
5. Copy the registration token (starts with `AHRA...`)

### Step 2: Store the Token Securely

Option A: File storage (recommended)
```bash
mkdir -p secrets
echo "YOUR_TOKEN_HERE" > secrets/github_runner_token
chmod 600 secrets/github_runner_token
```

Option B: Environment variable
```bash
export GITHUB_RUNNER_TOKEN="YOUR_TOKEN_HERE"
```

### Step 3: Register and Configure Runner

```bash
# Navigate to the githubagent directory
cd githubagent

# Register the runner (replace lee101/stock-prediction with your repo)
bash ./register_runner.sh \
  --url "https://github.com/lee101/stock-prediction" \
  --name "$(hostname)-runner" \
  --labels "stock-ci,linux,gpu" \
  --work "_work"
```

### Step 4: Choose Deployment Method

#### Option A: Run as systemd Service (Recommended)

```bash
# Install systemd service
sudo bash ./install_systemd.sh --name $(hostname)-runner

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable --now actions-runner@$(hostname)-runner

# Verify it's running
sudo systemctl status actions-runner@$(hostname)-runner
sudo journalctl -u actions-runner@$(hostname)-runner -f  # View logs
```

#### Option B: Run in Foreground (Testing)

```bash
cd actions-runner
./run.sh
# Press Ctrl+C to stop
```

### Step 5: Verify in GitHub

1. Go to **Settings** → **Actions** → **Runners**
2. Your runner should appear as "Idle" (green dot)
3. Test with a workflow targeting your runner labels

## Managing Runners

### Updating

Runners auto-update when idle. For manual updates:

```bash
# Stop the runner
sudo systemctl stop actions-runner@$(hostname)-runner

# Remove and re-register
cd githubagent
bash ./remove_runner.sh
bash ./register_runner.sh --url "https://github.com/lee101/stock-prediction" --name "$(hostname)-runner"

# Restart service
sudo systemctl start actions-runner@$(hostname)-runner
```

### Monitoring

```bash
# Check service status
sudo systemctl status actions-runner@*

# View logs
sudo journalctl -u actions-runner@$(hostname)-runner --since "1 hour ago"

# Check runner processes
ps aux | grep actions.runner
```

### Removing Runners

```bash
cd githubagent

# Set token and URL for removal
export GITHUB_RUNNER_TOKEN="YOUR_REMOVAL_TOKEN"
export GITHUB_RUNNER_URL="https://github.com/lee101/stock-prediction"

# Remove runner registration
bash ./remove_runner.sh

# If using systemd, clean up service
RUNNER_NAME=$(hostname)-runner
sudo systemctl disable --now actions-runner@${RUNNER_NAME}
sudo rm -f /etc/systemd/system/actions-runner@${RUNNER_NAME}.service
sudo systemctl daemon-reload

# Clean up runner directory (optional)
rm -rf actions-runner/
```

## Docker Deployment (Alternative)

### Setup

```bash
cd githubagent

# Create environment file from template
cp ci-runner.env.example ci-runner.env

# Edit configuration
cat > ci-runner.env <<EOF
GITHUB_RUNNER_URL=https://github.com/lee101/stock-prediction
GITHUB_RUNNER_TOKEN=YOUR_TOKEN_HERE
GITHUB_RUNNER_NAME=$(hostname)-docker-runner
GITHUB_RUNNER_LABELS=stock-ci,linux,gpu,docker
EOF

# Start container
docker compose up -d

# View logs
docker compose logs -f

# Stop and remove
docker compose down
```

## Using in Workflows

### Example Workflow

```yaml
name: CI on Self-Hosted
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: [self-hosted, stock-ci, gpu]  # Target your runner
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          pytest
```

### Available Workflows

- `.github/workflows/ci.yml` - Main CI workflow using this self-hosted runner
- Targets runners with label: `stock-ci`

### Security Best Practices

1. **Never hardcode secrets** - Use GitHub Secrets for sensitive data
2. **Restrict runner scope** - Use repository runners, not organization runners
3. **Regular updates** - Keep runner software updated
4. **Monitor access** - Review runner logs regularly
5. **Network isolation** - Consider running in isolated network if possible

## Troubleshooting

### Runner Not Starting

```bash
# Check service logs
sudo journalctl -u actions-runner@* -n 50

# Verify runner directory
ls -la githubagent/actions-runner/

# Test network connectivity
curl -I https://github.com
```

### Runner Offline in GitHub

1. Check runner service is running
2. Verify network connectivity
3. Check for expired tokens (re-register if needed)
4. Review firewall rules

### Permission Issues

```bash
# Fix ownership
sudo chown -R $USER:$USER githubagent/actions-runner/

# Fix permissions
chmod +x githubagent/*.sh
```

## Support

- GitHub Actions Documentation: https://docs.github.com/en/actions
- Runner Troubleshooting: https://docs.github.com/en/actions/hosting-your-own-runners/troubleshooting
