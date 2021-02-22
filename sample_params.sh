#!/bin/bash

# python -m tf_bodypix replace-background --source /dev/video0 --mask-mean-count=2 --mask-blur=10 --threshold=0.65 --background ~/Documents/art/beach.webp --output /dev/video4

python -m tf_bodypix auto-track --source /dev/video0 --mask-mean-count=2 --mask-blur=10 --threshold=0.65 \
--crops-mean-count=8 --padding=50 --output /dev/video4

# --source-width=1280 --source-height=960 
