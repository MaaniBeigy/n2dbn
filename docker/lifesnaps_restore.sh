#!/bin/sh
# Fetch the published LifeSnaps dump and restore it into MongoDB.
#
# The archive is 613 MiB and carries a 9.0 GiB `fitbit.bson`, so the restore
# needs about 12 GiB free under the mongo volume.  A restore that already ran
# is left alone unless FORCE is set.
set -eu

ZENODO_URL="https://zenodo.org/records/6832242/files/rais_anonymized.zip?download=1"
ARCHIVE_MD5="ada6e54d6b75845ed066122725643ef6"
WORK="${WORK:-/data/lifesnaps}"
DB="${MONGO_DB:-rais_anonymized}"
HOST="${MONGO_HOST:-mongo:27017}"

mkdir -p "$WORK"

if [ -z "${FORCE:-}" ] && mongosh --quiet --host "$HOST" --eval \
        "db.getSiblingDB('$DB').fitbit.countDocuments({}) > 0" 2>/dev/null | grep -q true; then
    echo "[lifesnaps] $DB already restored; set FORCE=1 to redo it"
    exit 0
fi

have() { command -v "$1" >/dev/null 2>&1; }

# The mongo image ships no downloader, so install one on first use.
if ! have curl && ! have wget && ! have python3 && ! have python; then
    echo "[lifesnaps] installing curl and unzip into the mongo image"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl unzip >/dev/null
fi

if [ ! -f "$WORK/rais_anonymized.zip" ]; then
    echo "[lifesnaps] downloading the published archive"
    if have curl; then
        curl -sSL -o "$WORK/rais_anonymized.zip" "$ZENODO_URL"
    elif have wget; then
        wget -q -O "$WORK/rais_anonymized.zip" "$ZENODO_URL"
    elif have python3; then
        python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[2], sys.argv[1])" \
            "$WORK/rais_anonymized.zip" "$ZENODO_URL"
    else
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[2], sys.argv[1])" \
            "$WORK/rais_anonymized.zip" "$ZENODO_URL"
    fi
fi

echo "[lifesnaps] checking the archive against its published digest"
echo "$ARCHIVE_MD5  $WORK/rais_anonymized.zip" | md5sum -c -

if [ ! -f "$WORK/mongo_rais_anonymized/sema.bson" ]; then
    echo "[lifesnaps] unpacking"
    if have unzip; then
        unzip -q "$WORK/rais_anonymized.zip" -d "$WORK"
    elif have python3; then
        python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
            "$WORK/rais_anonymized.zip" "$WORK"
    else
        python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
            "$WORK/rais_anonymized.zip" "$WORK"
    fi
fi

echo "[lifesnaps] restoring into $DB at $HOST"
mongorestore --host "$HOST" --db "$DB" --drop "$WORK/mongo_rais_anonymized"
echo "[lifesnaps] done"
