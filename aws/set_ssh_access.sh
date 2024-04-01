# init environment
python set_ssh_access.py --init

# generate ssh key pair
python set_ssh_access.py --generate-keys-master

# collect ssh keys
python set_ssh_access.py --set-keys-master
