set -e
set -x

sudo apt-get install python3-pip
sudo pip3 install virtualenv

virtualenv -p /usr/bin/python3.6 bsuiteenv
source bsuiteenv/bin/activate

git clone sso://team/dm-env-owners/dm_env
pip3 install dm_env/

git clone sso://team/deepmind-eng/bsuite
pip3 install bsuite/

pip3 install nose
nosetests bsuite/bsuite/tests/environments_test.py

python3 -c "import bsuite
env = bsuite.load_from_id('catch/0')
env.reset()"


deactivate
rm -rf bsuiteenv/
rm -rf bsuite/
rm -rf dm_env/
