#!/bin/bash
# call via nohup ./train_p2.sh 0 &

cd "$(dirname ${0})" || exit

gpu=$1
# See common/cfgs.sh for all options
cfg_training="seg"
cfg_frozen="frozen"
cfg_bias="all"
cfg_labels="-1"  # -1: ckpt from self-sup, 0: ckpt from all, 1: ckpt from part I, 2: ckpt from part II
cfg_downstream=true
cfg_dimensions="3d_new"
cfg_architecture="wip"
cfg_model_variant="punet_decoder"
cfg_pretrained="none"
cfg_prompting="full"
cfg_adaptation="prompting"
cfg_dataset="tcia_btcv"  # tcia_btcv or ctorg
cfg_amount="-1"  # use half the amount of annotated for tcia_btcv since those are two datasets
ckpts="none"  # "none" or (w&b) run name
username="marc"
labels_downstream=("1" "2" "3" "5")
misc="${*:2} --no_sim_legacy"  # fetch remaining parameters

# loop through runs
timestamp=$(date +%Y%m%d_%H%M%S)
multiple_out="multiple_${timestamp}.out"
multiple_err="multiple_${timestamp}.err"
for label_ in "${labels_downstream[@]}"; do
  sleep 2
  misc_="${misc} --label_indices_downstream_active ${label_}"
  echo "bash ./train_short.sh ${gpu} ${cfg_training} ${cfg_frozen} ${cfg_bias} ${cfg_labels} ${cfg_downstream} ${cfg_dimensions} ${cfg_architecture} ${cfg_model_variant} ${cfg_pretrained} ${cfg_prompting} ${cfg_adaptation} ${cfg_dataset} ${cfg_amount} ${ckpts} ${username} ${misc_} >> ${multiple_out} 2>> ${multiple_err} &" >> ${multiple_out} 2>> ${multiple_err}
  bash ./train_short.sh ${gpu} ${cfg_training} ${cfg_frozen} ${cfg_bias} ${cfg_labels} ${cfg_downstream} ${cfg_dimensions} ${cfg_architecture} ${cfg_model_variant} ${cfg_pretrained} ${cfg_prompting} ${cfg_adaptation} ${cfg_dataset} ${cfg_amount} ${ckpts} ${username} ${misc_} >> ${multiple_out} 2>> ${multiple_err} &
  wait
done
