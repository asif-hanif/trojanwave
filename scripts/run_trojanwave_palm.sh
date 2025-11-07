#!/bin/bash
METHOD=palm
for ATTACK in trojanwave;  do  sh scripts/beijing_opera.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/crema_d.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/esc50_actions.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/esc50.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/gt_music_genre.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/ns_instruments.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/ravdess.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/sesa.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/tut.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/urban_sound.sh $METHOD $ATTACK;   done
for ATTACK in trojanwave;  do  sh scripts/vocal_sound.sh $METHOD $ATTACK;   done