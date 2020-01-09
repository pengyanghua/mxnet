
rsync -r ./* --rsync-path="sudo rsync" net-g4:/home/net/eclipse-rse-mxnet/
ssh net-g4 << EOF
  cd eclipse-rse-mxnet
  sudo ./compile.sh
  sudo ./compile.sh
EOF
