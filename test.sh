set -e
set -x

cd ~/

rm -rf bsuite_env/
rm -rf bsuite/

sudo apt-get install python3-pip
sudo pip3 install virtualenv

virtualenv -p /usr/bin/python3.6 bsuite_env
source bsuite_env/bin/activate

git clone sso://team/deepmind-eng/bsuite
pip3 install bsuite/

pip3 install nose
nosetests bsuite/bsuite/tests/environments_test.py

python3 -c "import bsuite
env = bsuite.load_from_id('catch/0')
env.reset()"

deactivate
rm -rf bsuite_env/
rm -rf bsuite/
