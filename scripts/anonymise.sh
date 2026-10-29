#!/bin/zsh
# The anonymous code archive for double-blind review: dist/driftwatch-anonymous.zip.
#
# Packs the committed tree at HEAD (never the working copy), leaves out the
# paper, the team's work-split notes, the project brief, the CI config and this script, and refuses to finish
# if any team name, handle or repository link is left in what remains. The
# zip carries no git metadata: it is rebuilt from a plain extract, so the
# commit id git writes into its own archives is not in it.
set -euo pipefail
cd "${0:A:h}/.."

OUT=dist/driftwatch-anonymous.zip
EXCLUDE=(paper docs/WORKSTREAMS.md docs/PROJECT_BRIEF.md docs/PROJECT_BRIEF.pdf docs/DRIFTWATCH_DEEP_DIVE.md .github scripts/anonymise.sh)
PATTERN='dhruv|parmar|pronnati|tapaswi|mahak|khurdia|aditya|dhirubhai|daiict|github\.com/'

TMP=$(mktemp -d)
trap 'rm -rf $TMP' EXIT
git archive --format=tar --prefix=driftwatch/ HEAD | tar -x -C $TMP
for p in $EXCLUDE; do rm -rf $TMP/driftwatch/$p; done

# The hand-label sheets are named after the labellers: labeller_a-d in the
# archive, in the config's list and in the sheet file names alike.
CFG=$TMP/driftwatch/src/config/hand_label_study.yaml
LABELLERS=(${(s:, :)$(sed -n 's/^labellers: \[\(.*\)\]$/\1/p' $CFG)})
i=0
for name in $LABELLERS; do
  anon=labeller_$(printf "\\x$(printf %x $((97 + i)))")
  mv $TMP/driftwatch/data/hand_labels/sheet_$name.csv $TMP/driftwatch/data/hand_labels/sheet_$anon.csv
  i=$((i + 1))
done
sed -i '' 's/^labellers: \[.*\]$/labellers: [labeller_a, labeller_b, labeller_c, labeller_d]/' $CFG

if grep -r -I -i -n -E $PATTERN $TMP/driftwatch; then
  echo "identifying text left in the archive (above); nothing written" >&2
  exit 1
fi
if (cd $TMP && find driftwatch | grep -i -E $PATTERN); then
  echo "identifying file names left in the archive (above); nothing written" >&2
  exit 1
fi

mkdir -p dist
rm -f $OUT
(cd $TMP && zip -q -r -X - driftwatch) > $OUT
echo "$OUT: $(unzip -l $OUT | tail -1 | awk '{print $2}') files"
