#!/bin/bash
METHOD=palm
for ATTACK in flowmur;  do  sh scripts/beijing_opera.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/crema_d.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/esc50_actions.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/esc50.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/gt_music_genre.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/ns_instruments.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/ravdess.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/sesa.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/tut.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/urban_sound.sh $METHOD $ATTACK;   done
for ATTACK in flowmur;  do  sh scripts/vocal_sound.sh $METHOD $ATTACK;   done