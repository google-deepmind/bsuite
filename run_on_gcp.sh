#!/bin/bash
# Follow the instructions in README.md to setup Cloud SDK first.

# gcloud settings below
export IMAGE_FAMILY="tf-1-13-cpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="bsuite$RANDOM"
export MACHINE_TYPE="n1-highcpu-64" # or n1-highcpu-8 for debugging etc

# run settings below
export SCRIPT_TO_RUN="~/bsuite/bsuite/baselines/dqn/run.py"
export BSUITE_ENV="SWEEP"

SECONDS=0

set -e

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --machine-type=$MACHINE_TYPE

until gcloud compute ssh $INSTANCE_NAME --command "git clone \
https://github.com/deepmind/bsuite.git" --zone $ZONE &> /dev/null
do
  echo "Waiting for the instance to be initiated."
  sleep 10
done

gcloud compute ssh $INSTANCE_NAME --command "sudo apt-get install python3-pip" --zone $ZONE
gcloud compute ssh $INSTANCE_NAME --command "sudo pip3 install virtualenv" --zone $ZONE
gcloud compute ssh $INSTANCE_NAME --command "virtualenv -p /usr/bin/python3 bsuite_env"
gcloud compute ssh $INSTANCE_NAME --command "source ~/bsuite_env/bin/activate \
&& pip3 install ~/bsuite/[baselines]" --zone $ZONE

gcloud compute ssh $INSTANCE_NAME --command "nohup bash -c 'source \
~/bsuite_env/bin/activate && python3 $SCRIPT_TO_RUN \
--bsuite_id=$BSUITE_ENV --logging_mode=sqlite > /dev/null 2>&1 \
&& touch /tmp/bsuite_completed.txt > /dev/null 2>&1' 1>/dev/null \
2>/dev/null &" --zone $ZONE


until gcloud compute ssh $INSTANCE_NAME --command "cat /tmp/bsuite_completed.txt" \
--zone $ZONE &> /dev/null
do
  echo "Waiting for jobs to be completed."
  sleep 60
done

gcloud compute scp --recurse $INSTANCE_NAME:/tmp/bsuite.db /tmp/bsuite.db --zone $ZONE

echo "Experiments completed!"

gcloud compute instances stop $INSTANCE_NAME \
  --zone=$ZONE

echo "The experiment took $SECONDS seconds."
