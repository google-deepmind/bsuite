set -e
set -x

sudo apt-get install python3-pip
sudo pip3 install virtualenv

virtualenv -p /usr/bin/python3.6 ~/bsuite/venv
source ~/bsuite/venv/bin/activate

git clone sso://team/dm-env-owners/dm_env ~/bsuite/temp/dm_env
pip3 install ~/bsuite/temp/dm_env

git clone sso://team/deepmind-eng/bsuite ~/bsuite/temp/bsuite
pip3 install ~/bsuite/temp/bsuite

pip3 install nose
nosetests tests/environments_test.py

python3 -c "import bsuite
env = bsuite.load_from_id('catch/0')
env.reset()"


deactivate
rm -rf ~/bsuite/temp
