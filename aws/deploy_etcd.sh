RDZV_IP=${RDZV_IP:-127.0.0.1}
RDZV_PORT=${RDZV_PORT:-2134}

echo $RDZV_IP, $RDZV_PORT

mkdir -p /tmp/tmp_etcd
cd /tmp/tmp_etcd
rm -rf default.etcd
cmd="etcd --enable-v2 \
    --listen-client-urls http://0.0.0.0:${RDZV_PORT},http://127.0.0.1:4222 \
    --advertise-client-urls http://${RDZV_IP}:${RDZV_PORT}"

echo ${cmd}
eval ${cmd}

set +x
