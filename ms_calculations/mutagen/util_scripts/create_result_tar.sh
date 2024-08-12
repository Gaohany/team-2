#/bin/bash
#tar --exclude 'checkpoint.tar' --exclude '*__pycache__*' --sort=name --mtime="@0" --owner=0 --group=0 --numeric-owner  --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime -hcvf stn_evaln_${1}.tar results/stn/evaln/dataset\=${1}/
tar --exclude '' --exclude 'checkpoint.tar' --exclude '*__pycache__*' --sort=name --mtime="@0" --owner=0 --group=0 --numeric-owner  --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime -hzcvf to_upload/stn_evaln_fair_${1}.tar.gz results/stn/evaln_fair/dataset\=${1}/
