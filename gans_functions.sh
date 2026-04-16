#!/bin/bash

# This script runs the GANS training of  unsupervised wav2vec pipeline

# Wav2Vec Unsupervised Pipeline
# with checkpointing to allow resuming from any step

set -e
set -o pipefail

source "$(dirname "$0")/utils.sh"

#=========================== GANS training and preparation ==============================
train_gans(){
   local step_name="train_gans"
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH="${DIR_PATH}:${FAIRSEQ_ROOT}:${PYTHONPATH:-}"


   if is_completed "$step_name"; then
        log "Skipping gans training  (already completed)"
        return 0
    fi

    log "gans training."
    mark_in_progress "$step_name"

   # SANITY_GAN=1: short smoke test (400 updates). Uses 50% train audio.
   if [ "${SANITY_GAN:-0}" = "1" ]; then
        log "SANITY_GAN=1: single run, max_update=400, train_audio_subsample_ratio=0.5"
        PYTHONPATH="${DIR_PATH}:${FAIRSEQ_ROOT}:${PYTHONPATH:-}" PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
            --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
            --config-name w2vu \
            task.train_audio_subsample_ratio=0.5 \
            task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
            task.text_data="$TEXT_OUTPUT/phones/" \
            task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
            common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
            common.seed=0 \
            model.input_dim=1024 \
            model.discriminator_dim=256 model.gradient_penalty=2.0 model.code_penalty=6.0 \
            model.smoothness_weight=2.0 model.smoothing=0.1 model.discriminator_dropout=0.1 \
            optimization.max_update=400 checkpoint.save_interval_updates=100 \
            optimization.clip_norm=0.5 \
            dataset.validate_interval_updates=100 \
            common.fp16=false \
            +optimizer.groups.generator.optimizer.lr="[0.00002]" \
            +optimizer.groups.discriminator.optimizer.lr="[0.00001]" \
            ~optimizer.groups.generator.optimizer.amsgrad \
            ~optimizer.groups.discriminator.optimizer.amsgrad \
            2>&1 | tee $RESULTS_DIR/training1.log
   else
   # Single run; defaults from w2vu.yaml (50% train audio, max_update in yaml).
   PYTHONPATH="${DIR_PATH}:${FAIRSEQ_ROOT}:${PYTHONPATH:-}" PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
    --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
    task.text_data="$TEXT_OUTPUT/phones/" \
    task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    common.seed=0 \
    model.input_dim=1024 \
    model.discriminator_dim=256 model.gradient_penalty=2.0 model.code_penalty=6.0 \
    model.smoothness_weight=2.0 model.smoothing=0.1 model.discriminator_dropout=0.1 \
    optimization.clip_norm=0.5 \
    common.fp16=false \
    +optimizer.groups.generator.optimizer.lr="[0.00002]" \
    +optimizer.groups.discriminator.optimizer.lr="[0.00001]" \
    ~optimizer.groups.generator.optimizer.amsgrad \
    ~optimizer.groups.discriminator.optimizer.amsgrad \
    2>&1 | tee $RESULTS_DIR/training1.log
   fi

   if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "gans trained successfully"
    else
        log "ERROR: gans training failed"
        exit 1
    fi
}
