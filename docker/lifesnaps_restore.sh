#!/bin/sh
# Fetch the published LifeSnaps dump and restore it into MongoDB.
#
# The archive is 613 MiB and carries a 9.0 GiB `fitbit.bson`, so the restore
# needs about 12 GiB free under the mongo volume.  A restore that already ran
# is left alone unless FORCE is set.
#
# Download source: the API content endpoint of one pinned Zenodo record
# (6832242).  Do NOT rely on the /records/<id>/files/...?download=1 HTML
# route: after Zenodo's migration that route can follow redirects to the
# newest version of the concept, whose zip has a different checksum.
# Zenodo's API also rejects curl's default User-Agent with 403 on some
# networks, so every request sends a descriptive UA and no custom Accept
# header (a custom Accept: application/octet-stream gets 406).
set -eu

ZENODO_RECORD="${ZENODO_RECORD:-6832242}"
ARCHIVE_NAME="rais_anonymized.zip"
ZENODO_URL="https://zenodo.org/api/records/${ZENODO_RECORD}/files/${ARCHIVE_NAME}/content"
ZENODO_HTML_URL="https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE_NAME}?download=1"
UA="n2dbn-lifesnaps-restore/1.0 (WSDM N2DBN revision; contact: project maintainers)"
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

# api_get <url> [output-file] : fetch with a descriptive UA, no custom Accept.
api_get() {
    if have curl; then
        if [ $# -ge 2 ]; then
            curl -fL --retry 3 --retry-delay 5 -A "$UA" -o "$2" "$1"
        else
            curl -fsSL --retry 3 --retry-delay 5 -A "$UA" "$1"
        fi
    elif have wget; then
        if [ $# -ge 2 ]; then
            wget --user-agent="$UA" --retry-connrefused --tries=3 -O "$2" "$1"
        else
            wget -q --user-agent="$UA" -O - "$1"
        fi
    elif have python3; then
        python3 -c "import sys, urllib.request
req = urllib.request.Request(sys.argv[1], headers={'User-Agent': sys.argv[2]})
data = urllib.request.urlopen(req).read()
if len(sys.argv) > 3:
    open(sys.argv[3], 'wb').write(data)
else:
    sys.stdout.buffer.write(data)" "$1" "$UA" ${2:+"$2"}
    else
        python -c "import sys, urllib.request
req = urllib.request.Request(sys.argv[1], headers={'User-Agent': sys.argv[2]})
data = urllib.request.urlopen(req).read()
if len(sys.argv) > 3:
    open(sys.argv[3], 'wb').write(data)
else:
    sys.stdout.buffer.write(data)" "$1" "$UA" ${2:+"$2"}
    fi
}

# published_md5 : the md5 Zenodo lists for this file in this record's JSON.
published_md5() {
    api_get "https://zenodo.org/api/records/${ZENODO_RECORD}" 2>/dev/null \
        | tr '{' '\n' | grep "$ARCHIVE_NAME" \
        | grep -o '"checksum": *"md5:[0-9a-f]*"' \
        | grep -o '[0-9a-f]\{32\}' | head -1
}

download() {
    if api_get "$ZENODO_URL" "$WORK/$ARCHIVE_NAME"; then
        return 0
    fi
    echo "[lifesnaps] API content endpoint failed; falling back to the HTML download route"
    echo "[lifesnaps] (that route may serve a newer version; the digest check below catches it)"
    api_get "$ZENODO_HTML_URL" "$WORK/$ARCHIVE_NAME"
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
        published=$(published_md5) || published=""
        rm -f "$WORK/$ARCHIVE_NAME"
        echo "[lifesnaps] digest mismatch: got $actual, pinned $ARCHIVE_MD5, Zenodo-published ${published:-unknown}" >&2
        echo "[lifesnaps] the archive was removed; if Zenodo-published differs from pinned, the record" >&2
        echo "[lifesnaps] now serves a different file: set ZENODO_RECORD/ARCHIVE_MD5 accordingly, then rerun" >&2
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
