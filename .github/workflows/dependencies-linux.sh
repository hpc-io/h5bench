set -eu -o pipefail

sudo apt-get update

# mpi
sudo apt-get install libopenmpi-dev

# hdf5
sudo apt-get install libhdf5-serial-dev
