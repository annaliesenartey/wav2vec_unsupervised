#!/bin/bash

# shellcheck source=gans_functions.sh
source "$(dirname "$0")/gans_functions.sh"
# So manual "sed ... progress.checkpoint" matches CLUSTERING_DIR ($DATASET_NAME); see utils.sh
mkdir -p "$CHECKPOINT_DIR"
[ -f "$CHECKPOINT_FILE" ] || touch "$CHECKPOINT_FILE"

activate_venv  
setup_path 
create_dirs #creates directories for storing outputs from the different steps 

train_gans

log "Pipeline completed successfully!"
