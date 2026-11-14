#!/bin/sh
# Fetch the published LifeSnaps dump and restore it into MongoDB.
#
# The archive is 613 MiB and carries a 9.0 GiB `fitbit.bson`, so the restore
# needs about 12 GiB free under the mongo volume.  A restore that already ran
# is left alone unless FORCE is set.
#
# The download URL is pinned to the API content endpoint of one specific Zenodo
# record (6832242). Do NOT use the /records/<id>/files/...?download=1 HTML
# route: after Zenodo's migration that route follows redirects to the newest
# version of the concept, whose zip has a different checksum, so the pinned
# md5 check fails even though the download itself is intact.
set -eu

ZENODO_RECORD="${ZENODO_RECORD:-6832242}"
ARCHIVE_NAME="rais_anonymized.zip"
ZENODO_URL="https://zenodo.org/api/records/${ZENODO_RECORD}/files/${ARCHIVE_NAME}/content"
# Checksum and size published by Zenodo for this exact record and file.
ARCHIVE_MD5="ada6e54d6b75845ed066122725643ef6"
ARCHIVE_SIZE=642961582
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

download() {
    if have curl; then
        curl -fL --retry 3 --retry-delay 5 -o "$WORK/$ARCHIVE_NAME" "$ZENODO_URL"
    elif have wget; then
        wget --retry-connrefused --tries=3 -O "$WORK/$ARCHIVE_NAME" "$ZENODO_URL"
    elif have python3; then
        python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[2], sys.argv[1])" \
            "$WORK/$ARCHIVE_NAME" "$ZENODO_URL"
    else
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[2], sys.argv[1])" \
            "$WORK/$ARCHIVE_NAME" "$ZENODO_URL"
    fi
}

verified() {
    [ -f "$1" ] || return 1
    echo "$ARCHIVE_MD5  $1" | md5sum -c - >/dev/null 2>&1
}

if verified "$WORK/$ARCHIVE_NAME"; then
    echo "[lifesnaps] archive already present and digest-verified"
else
    # A previous failed attempt must not block the re-download.
    rm -f "$WORK/$ARCHIVE_NAME"
    echo "[lifesnaps] downloading Zenodo record $ZENODO_RECORD (expect $ARCHIVE_SIZE bytes)"
    download
    size=$(stat -c %s "$WORK/$ARCHIVE_NAME")
    echo "[lifesnaps] downloaded $size bytes; checking the digest"
    if ! verified "$WORK/$ARCHIVE_NAME"; then
        actual=$(md5sum "$WORK/$ARCHIVE_NAME" | cut -d' ' -f1)
        published=""
        if have curl; then
            published=$(curl -fsSL "https://zenodo.org/api/records/${ZENODO_RECORD}" 2>/dev/null \
                | sed -n "s/.*\"key\": *\"${ARCHIVE_NAME}\".*\"checksum\": *\"md5:\([0-9a-f]*\)\".*/\1/p" \
                | head -1) || published=""
        fi
        rm -f "$WORK/$ARCHIVE_NAME"
        echo "[lifesnaps] digest mismatch: got $actual, pinned $ARCHIVE_MD5, Zenodo-published ${published:-unknown}" >&2
        echo "[lifesnaps] the archive was removed; set ZENODO_RECORD to a record whose published" >&2
        echo "[lifesnaps] checksum matches, or update ARCHIVE_MD5/ARCHIVE_SIZE, then rerun" >&2
        exit 1
    fi
fi

echo "[lifesnaps] archive digest verified"

if [ ! -f "$WORK/mongo_rais_anonymized/sema.bson" ]; then
    echo "[lifesnaps] unpacking"
    if have unzip; then
        unzip -q "$WORK/$ARCHIVE_NAME" -d "$WORK"
    elif have python3; then
        python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
            "$WORK/$ARCHIVE_NAME" "$WORK"
    else
        python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
            "$WORK/$ARCHIVE_NAME" "$WORK"
    fi
fi

echo "[lifesnaps] restoring into $DB at $HOST"
mongorestore --host "$HOST" --db "$DB" --drop "$WORK/mongo_rais_anonymized"
echo "[lifesnaps] done"
