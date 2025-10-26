# Remote GPU Provisioning & Experiment Workflow

This document captures the end-to-end flow for running the market simulator on a rented GPU (e.g., a Vast.ai RTX 5090 instance). It includes the container build/push process, expected environment configuration on the remote machine, and a non-docker helper script for synchronising datasets and experiment logs with the Cloudflare R2 bucket.

## Prerequisites

- `VAST_API_KEY` exported locally (`source ~/.secretbashrc` already exposes it).
- `docker` and `uv` installed on the build host.
- Credentials for the private container registry you plan to push to (GHCR, Docker Hub, etc.).
- Cloudflare R2 access keys stored as environment variables (see below).
- AWS CLI available both locally and on the remote machine for `aws s3 sync` commands.

## Build & Push the Simulator Image

The script `marketsimulator/scripts/build_and_push_simulator.sh` builds `marketsimulator/Dockerfile` from the repo root and pushes it to the registry of your choice.

```bash
export REGISTRY_URL=ghcr.io/your-org            # or private registry URL
export IMAGE_NAME=marketsimulator              # override as needed
export IMAGE_TAG=sim-$(date -u +%Y%m%d-%H%M%S)  # optional explicit tag
export REGISTRY_USERNAME=ghcr-user             # optional, only if login is required
export REGISTRY_PASSWORD=ghcr-token            # optional, only if login is required

./marketsimulator/scripts/build_and_push_simulator.sh
```

The `marketsimulator/Dockerfile` installs the repository via `uv` inside a CUDA 12.4 runtime image, sets up the expected data directories (`trainingdata*`, `compiled_models`, `hyperparams`), and defaults to running `python -m marketsimulator.runner`. Adjust `DOCKER_BUILD_ARGS` (e.g., `--build-arg CUDA_VERSION=12.5.0`) when you need a different CUDA stack.

## Launching a Vast.ai RTX 5090 Instance

Once sufficient credit is available on the Vast.ai account:

1. Search for a verified RTX 5090 machine and note the offer ID, e.g. `27097509` in Texas.
2. Launch an SSH-capable instance with the freshly pushed image:

   ```bash
   source ~/.secretbashrc
   vastai create instance 27097509 \
       --image ghcr.io/your-org/marketsimulator:sim-20251025-1200 \
       --disk 200 \
       --ssh \
       --direct \
       --label marketsimulator-rtx5090
   ```

3. Poll until the instance is running, then fetch the SSH command:

   ```bash
   vastai show instances | grep marketsimulator-rtx5090
   vastai ssh-url <contract_id>
   # or: vastai attach ssh <contract_id>
   ```

4. After connecting, run container workloads using `docker run --gpus all`. For long simulations mount persistent volumes or bind mounts as required.

## Non-Docker Workflow: Dataset & Log Sync

On a bare-metal session (or inside the container) use `marketsimulator/scripts/sync_experiment_assets.sh` to pull required datasets and push logs to R2.

### Environment Variables

```bash
export R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"
export R2_BUCKET="models"                           # bucket that contains the stock folder
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export EXPERIMENT_ID="marketsim-rtx5090-test"       # optional; auto-generated if omitted
export LOCAL_ROOT="/workspace"                      # optional; defaults to /workspace
export LOCAL_LOG_DIR="marketsimulator/run_logs"     # optional override
export EXTRA_PULL_DIRS="compiled_models_aux,results" # optional CSV list of extra subdirs
```

### Sync + Run Example

```bash
./marketsimulator/scripts/sync_experiment_assets.sh \
    --run "python -m marketsimulator.runner --config configs/live.yaml"
```

The script performs three stages:

1. Pulls `trainingdata/`, `trainingdatadaily/`, `trainingdatahourly/`, `compiled_models/`, and `hyperparams/` from `s3://$R2_BUCKET/models/stock/` via the provided R2 endpoint. Extra directories can be mirrored by listing them in `EXTRA_PULL_DIRS`.
2. Executes the optional experiment command (wrapped in `bash -lc` so shell features work). If no command is supplied the script only performs the syncs.
3. Syncs `${LOCAL_ROOT}/${LOCAL_LOG_DIR}` back to `s3://$R2_BUCKET/models/stock/marketsimulator/logs/$EXPERIMENT_ID/`, ensuring logs upload even when the experiment command fails.

For ad-hoc transfers you can still run manual commands, e.g.:

```bash
aws s3 sync trainingdata/ s3://models/stock/trainingdata/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync marketsimulator/run_logs/ s3://models/stock/marketsimulator/logs/manual-20251025/ --endpoint-url "$R2_ENDPOINT"
```

## Checklist Before Running Experiments

- [ ] `vastai` CLI configured (`vastai show user` should report a positive balance).
- [ ] Container image pushed to the private registry and pullable from the remote host.
- [ ] AWS CLI installed on the instance (`sudo apt-get install awscli` if needed).
- [ ] `R2_*` credentials exported (consider using an `.env` file and `direnv`/`dotenv` to load them).
- [ ] `sync_experiment_assets.sh` run once to hydrate local datasets before starting long trainings or simulations.
- [ ] After the run, confirm that logs landed in `s3://models/stock/marketsimulator/logs/$EXPERIMENT_ID/`.

Following this flow makes it straightforward to spin up fresh GPUs, hydrate them with the necessary datasets, run the simulator, and persist experiment artefacts for later analysis.
